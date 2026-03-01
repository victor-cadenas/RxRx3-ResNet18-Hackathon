from src.dataloader import get_dataloaders, get_splits, preprocess
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.heatmap import generate_heatmap
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    ## Hyperparameters
    EPOCHS = 5
    BATCH_SIZE = 32
    LR = 3e-4
    CROP = 224
    NUM_WORKERS = 2
    SEED = 42

    set_seed(SEED)

     ## device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, val, test, labels_list = get_splits(seed=SEED)
    preprocess(train, val, test, crop_size=CROP)

    train_loader, val_loader, test_loader = get_dataloaders(
        train, val, test, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        crop_size=CROP
    )
    
    model, loss_fn, optimizer = get_model(labels_list, device, lr=LR)

    train_model(model, train_loader, val_loader, loss_fn, optimizer, device, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    evaluate_model(model, test_loader, device)

    generate_heatmap(model, train_loader, device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
