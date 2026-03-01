from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms


def get_splits(seed):
    #
    # =========================
    # 1) Download CSV metadata and create well sets (Vinicristine vs EMPTY_control)
    # =========================
    csv_path = hf_hub_download(
        "recursionpharma/rxrx3-core",
        filename="metadata_rxrx3_core.csv",
        repo_type="dataset"
    )

    df_meta = pd.read_csv(csv_path)

    vinicristine_df = df_meta[df_meta["treatment"] == "Vincristine"]
    vinicristine_wells = set(zip(
        vinicristine_df["experiment_name"],
        vinicristine_df["plate"],
        vinicristine_df["address"]
    ))

    empty_df = df_meta[df_meta["treatment"] == "EMPTY_control"]

    empty_df = empty_df.sample(n=min(len(empty_df), len(vinicristine_df)), random_state=seed) # Para balancear clases
    empty_wells = set(zip(
        empty_df["experiment_name"],
        empty_df["plate"],
        empty_df["address"]
    ))

    wanted_wells = vinicristine_wells.union(empty_wells)

    # =========================
    # 2) Download HF dataset (images + __key__)
    # =========================
    dataset = load_dataset("recursionpharma/rxrx3-core", split="train")

    # =========================
    # 3) Build a mapping for __key__ -> (exp, plate, well) -> label
    # =========================
    # Drop jp2 for a faster parse to pandas
    meta_only = dataset.remove_columns("jp2")
    df_keys = meta_only.to_pandas()

    # __key__ is like: "experiment/Plate1/B02_something"
    split_cols = df_keys["__key__"].str.split("/", expand=True)

    df_keys["exp"] = split_cols[0].astype(str)

    df_keys["plate"] = split_cols[1].astype(str)                      # "Plate1"
    df_keys["plate_num"] = df_keys["plate"].str.replace("Plate", "", regex=False).astype(int)

    df_keys["well"] = split_cols[2].str.split("_").str[0].astype(str) # "B02"

    # triple = (experiment_name, plate, address)
    df_keys["triple"] = list(zip(df_keys["exp"], df_keys["plate_num"], df_keys["well"]))

    # Create label:
    # - 1 if label in vinicristine_wells
    # - 0 if label in empty_wells
    # - None otherwise
    def get_label(triple):
        if triple in vinicristine_wells:
            return 1
        if triple in empty_wells:
            return 0
        return None

    df_keys["label"] = df_keys["triple"].apply(get_label)

    # Filter with labels
    df_filtered = df_keys[df_keys["label"].notna()].copy()

    indices = df_filtered.index.tolist()
    labels_list = df_filtered["label"].astype(int).tolist()

    # Select subset from dataset and add "label" column 
    filtered_dataset = dataset.select(indices)
    filtered_dataset = filtered_dataset.add_column("label", labels_list)

    # =========================
    # 4) Split Train/Val/Test (70%-15%-15%)
    # =========================
    splits = filtered_dataset.train_test_split(
        test_size=0.3,
        seed=seed,
        shuffle=True
    )

    train = splits["train"]
    temp = splits["test"]  

    # Split 30% into 15% val and 15% test
    splits_temp = temp.train_test_split(
        test_size=0.5,
        seed=seed,
        shuffle=True
    )

    val = splits_temp["train"]
    test = splits_temp["test"]

    return train, val, test, labels_list

def preprocess(train, val, test, crop_size):

    # =========================
    # 5) Transform: 1 image 512x512 -> 4 crops 224x224 
    # =========================

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    def transform_example(batch):
        
        # Convert image to tensor (C,H,W)

        images = batch["jp2"]
        all_patches = []

        for img in images:
            img = transforms.ToTensor()(img)  # (1,H,W)

            # if 1 channel image -> convert to 3 channel image
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            img = normalize(img)

            _, H, W = img.shape

            # 4 crops 224x224
            patches = torch.stack([
                img[:, 0:crop_size, 0:crop_size],        # up-left
                img[:, 0:crop_size, W-crop_size:W],      # up-right
                img[:, H-crop_size:H, 0:crop_size],      # down-left
                img[:, H-crop_size:H, W-crop_size:W],    # down-right
            ])
            # (4,C,224,224)
            all_patches.append(patches)

        # Save crops
        batch["pixel_values"] = all_patches  # (4,C,224,224)
        return batch

    train.set_transform(transform_example)
    val.set_transform(transform_example)
    test.set_transform(transform_example)


# =========================
# 6) Collate: convert batch of dicts -> (X, y) 
# =========================
def _make_collate_fn(crop_size):
    def collate_fn(batch):
        # Stack: (B,4,C,224,224)
        pixel_values = torch.stack([b["pixel_values"] for b in batch])

        # Calculate number of channels (just in case it's not RGB)
        C = pixel_values.shape[2]

        # Flatten crops in batch: (B*4,C,crop_size,crop_size)
        X = pixel_values.reshape(-1, C, crop_size, crop_size)

        # Labels: (B,) -> repeat 4 times -> (B*4,)
        y = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
        y = y.repeat_interleave(4)

        return X, y
    return collate_fn

# =========================
# 8) DataLoaders 
# =========================
def get_dataloaders(train, val, test, batch_size, num_workers, crop_size):
    collate = _make_collate_fn(crop_size)

    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate
    )

    val_loader = DataLoader(
        dataset=val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate
    )

    test_loader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate
    )

    return train_loader, val_loader, test_loader