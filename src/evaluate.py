import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def test_loop(model, test_loader, device):
  model.eval()

  all_preds = []
  all_labels = []

  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      _, preds = torch.max(outputs, 1)

      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

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

def evaluate_model(model, test_loader, device):
   
  final_metrics, cm = test_loop(model, test_loader, device)

  print(f"Accuracy:  {final_metrics['accuracy']*100:.1f}%")
  print(f"Precision: {final_metrics['precision']*100:.1f}%")
  print(f"Recall:    {final_metrics['recall']*100:.1f}%")
  print(f"F1 Score:  {final_metrics['f1']*100:.1f}%")

  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ineffective", "Effective"],
            yticklabels=["Ineffective", "Effective"])

  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Confusion Matrix")
  plt.show()

  return final_metrics, cm