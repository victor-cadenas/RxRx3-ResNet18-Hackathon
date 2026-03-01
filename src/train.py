import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score


def train_loop(model, dataloader, loss_fn, optimizer, device, batch_size):
    # Each image generates 4 crops -> multiply by 4
    train_size = len(dataloader.dataset) * 4
    nlotes = len(dataloader)

    model.train()

    perdida_train, exactitud = 0.0, 0.0

    for nlote, (X, y) in enumerate(dataloader):
        # Move to GPU/CPU
        X, y = X.to(device), y.to(device)

        # Forward
        logits = model(X)

        # Loss + backward
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Accuracy and loss calculation
        perdida_train += loss.item()
        exactitud += (logits.argmax(1) == y).type(torch.float).sum().item()

        # Print every 10 batches
        if nlote % 10 == 0:

            ndatos = nlote * batch_size * 4  # *4 because of crops
            print(f"\tPérdida: {loss.item():>7f}  [{ndatos:>5d}/{train_size:>5d}]")

    perdida_train /= max(nlotes, 1)
    exactitud /= max(train_size, 1)

    print(f"\tExactitud/pérdida promedio:")
    print(f"\t\tEntrenamiento: {(100*exactitud):>0.1f}% / {perdida_train:>8f}")

def val_loop(model, dataloader, loss_fn, device):
    # *4 because of crops
    val_size = len(dataloader.dataset) * 4
    nlotes = len(dataloader)

    model.eval()

    perdida_val, exactitud = 0.0, 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)

            _,preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            

            perdida_val += loss_fn(logits, y).item()
            exactitud += (logits.argmax(1) == y).type(torch.float).sum().item()
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    final_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
    }

    return final_metrics, cm

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, batch_size, epochs):

    for t in range(epochs):
        print(f"Iteration {t+1}/{epochs}\n-------------------------------")
        train_loop(model, train_loader, loss_fn, optimizer, device, batch_size)
        val_metrics, val_cm = val_loop(model, val_loader, loss_fn, device)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, f"checkpoint_epoch{epochs}.pth")