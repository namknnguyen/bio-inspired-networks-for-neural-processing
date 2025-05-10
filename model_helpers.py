import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np

def train_step(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(train_loader.dataset)

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds, zero_division=0)
    specificity = recall_score(1 - np.array(all_labels), 1 - np.array(all_preds), zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        'loss': total_loss / len(data_loader.dataset),
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1
    }