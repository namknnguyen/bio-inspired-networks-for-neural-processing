import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Dataset
from model_architectures import LSTM, CfcClassifier, SnnClassifier, LsmClassifier, EsnClassifier, SpikingGNNClassifier
from model_helpers import train_step, evaluate_model
import matplotlib.pyplot as plt
import logging
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="The verbose parameter is deprecated")

logging.basicConfig(filename='train_lstm.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
N_FOLDS = 10
INPUT_SIZE = 23
MODEL_TYPE = 'All'
PATIENT = 'chb07'

HYPERPARAMS = {
    'LSTM': {
        'HIDDEN_SIZE': 96,
        'DENSE_SIZE': 64,
        'DROPOUT': 0.1,
        'INPUT_PROJECTION_SIZE': 46,
    },
    'CfcClassifier': {
        'HIDDEN_SIZE1': 128,
        'DROPOUT': 0.1,
        'INPUT_PROJECTION_SIZE': 46,
        'DENSE_SIZE': 64,
        'CFC_HPARAMS': {
            "backbone_units": 64,
            "backbone_layers": 1,
            "backbone_activation": "relu",
            "backbone_dr": 0.1,
            "no_gate": False,
            "minimal": False,
            "init": 1.0
        },
    },
    'SnnClassifier': {
        'HIDDEN_SIZE1': 256,
        'DROPOUT': 0.1,
        'INPUT_PROJECTION_SIZE': 92,
        'DENSE_SIZE': 64,
        'SNN_HPARAMS': {
            "num_steps": 10,
            "beta": 0.95,
            "threshold": 0.75
        },
    },
    'LsmClassifier': {
        'HIDDEN_SIZE1': 256,
        'DROPOUT': 0.1,
        'INPUT_PROJECTION_SIZE': 46,
        'DENSE_SIZE': 128,
        'LSM_HPARAMS': {
            "num_steps": 5,
            "beta": 0.95,
            "threshold": 1.0,
            "spectral_radius": 0.9,
            "sparsity": 0.1
        },
    },
    'EsnClassifier': {
        'HIDDEN_SIZE1': 512,
        'DROPOUT': 0.1,
        'INPUT_PROJECTION_SIZE': 46,
        'DENSE_SIZE': 64,
        'ESN_HPARAMS': {
            "num_steps": 10,
            "spectral_radius": 0.9,
            "sparsity": 0.1,
            "leaking_rate": 0.5,
            "input_scaling": 1.0
        },
    },
    'SpikingGNNClassifier': {
        'HIDDEN_SIZE1': 128,
        'DROPOUT': 0.1,
        'INPUT_PROJECTION_SIZE': 92,
        'DENSE_SIZE': 64,
        'SPIKING_GNN_HPARAMS': {
            "num_steps": 5,
            "beta": 0.95,
            "threshold": 1.0,
            "num_nodes": 10
        },
    }
}

MODEL_REGISTRY = {
    'LSTM': LSTM,
    'CfcClassifier': CfcClassifier,
    'SnnClassifier': SnnClassifier,
    'LsmClassifier': LsmClassifier,
    'EsnClassifier': EsnClassifier,
    'SpikingGNNClassifier': SpikingGNNClassifier,
}

NOISE_STD = 0.1
SCALE_RANGE = (0.5, 1.5)
MAX_SHIFT = 4

def augment_data(data):
    batch_size, seq_len, input_size = data.shape
    augmented_data = data.clone()
    noise = torch.randn_like(augmented_data) * NOISE_STD
    augmented_data += noise
    scale_factors = torch.FloatTensor(batch_size).uniform_(*SCALE_RANGE).view(batch_size, 1, 1)
    augmented_data *= scale_factors
    shifts = torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (batch_size,))
    for i in range(batch_size):
        shift = shifts[i].item()
        augmented_data[i] = torch.roll(augmented_data[i], shifts=shift, dims=0)
    return augmented_data

class AugmentedTensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        augmented_data = augment_data(data.unsqueeze(0)).squeeze(0)
        return augmented_data, label

def load_data(data_dir, patient_id=PATIENT, fold_id='fold1'):
    train_pt = os.path.join(data_dir, f'train_{fold_id}_{patient_id}.pt')
    train_csv = os.path.join(data_dir, f'train_{fold_id}_{patient_id}.csv')
    test_pt = os.path.join(data_dir, f'test_{fold_id}_{patient_id}.pt')
    test_csv = os.path.join(data_dir, f'test_{fold_id}_{patient_id}.csv')
    train_data_dict = torch.load(train_pt)
    test_data_dict = torch.load(test_pt)
    train_data = train_data_dict['features']
    train_labels = train_data_dict['labels'].float()
    test_data = test_data_dict['features']
    test_labels = test_data_dict['labels'].float()
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    logging.info(f'Train CSV labels match: {np.all(train_df["label"].values == train_labels.numpy())}')
    logging.info(f'Test CSV labels match: {np.all(test_df["label"].values == test_labels.numpy())}')
    logging.info(f'Before balancing - Train: {train_data.shape[0]} segments, '
                 f'Pre-ictal: {(train_labels == 1).sum().item()}, '
                 f'Inter-ictal: {(train_labels == 0).sum().item()}')
    pre_ictal_idx = np.where(train_labels == 1)[0]
    inter_ictal_idx = np.where(train_labels == 0)[0]
    n_pre_ictal = len(pre_ictal_idx)
    if n_pre_ictal > 0 and len(inter_ictal_idx) > n_pre_ictal:
        inter_ictal_sampled = np.random.choice(inter_ictal_idx, n_pre_ictal, replace=False)
        balanced_idx = np.concatenate([pre_ictal_idx, inter_ictal_sampled])
        train_data = train_data[balanced_idx]
        train_labels = train_labels[balanced_idx]
        logging.info(f'After balancing - Train: {train_data.shape[0]} segments, '
                     f'Pre-ictal: {(train_labels == 1).sum().item()}, '
                     f'Inter-ictal: {(train_labels == 0).sum().item()}')
    else:
        logging.warning('Not enough segments to balance training data.')
    return train_data, train_labels, test_data, test_labels

def plot_metrics(fold_metrics, output_dir, identifier):
    epochs = range(1, EPOCHS + 1)
    mean_train_losses = np.mean([m['train_losses'] for m in fold_metrics], axis=0)
    mean_val_losses = np.mean([m['val_losses'] for m in fold_metrics], axis=0)
    mean_train_accuracies = np.mean([m['train_accuracies'] for m in fold_metrics], axis=0)
    mean_val_accuracies = np.mean([m['val_accuracies'] for m in fold_metrics], axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, mean_train_losses, label='Train Loss')
    plt.plot(epochs, mean_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - {identifier}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'loss_curve_{identifier}.png'))
    plt.close()
    logging.info(f'Saved loss curve for {identifier}')
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, mean_train_accuracies, label='Train Accuracy')
    plt.plot(epochs, mean_val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curve - {identifier}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'accuracy_curve_{identifier}.png'))
    plt.close()
    logging.info(f'Saved accuracy curve for {identifier}')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    data_dir = 'output2/'+PATIENT
    output_dir = './plots'
    patient_id = PATIENT
    fold_id = 'fold1'
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f'Loading data for {patient_id}...')
    train_data, train_labels, test_data, test_labels = load_data(data_dir, patient_id, fold_id)
    logging.info(f'Train data shape: {train_data.shape}, Test data shape: {test_data.shape}')
    
    if MODEL_TYPE == 'All':
        model_types = ['LSTM', 'CfcClassifier', 'SnnClassifier', 'LsmClassifier', 'EsnClassifier', 'SpikingGNNClassifier']
    else:
        model_types = [MODEL_TYPE]

    all_model_metrics = {}
    
    for model_type in model_types:
        logging.info(f'----- Starting training for model: {model_type} -----')
        fold_metrics = []
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(train_data), total=N_FOLDS, desc=f"Folds ({model_type})")):
            logging.info(f'Starting fold {fold + 1}/{N_FOLDS} for {model_type}')
            train_fold_data, val_data = train_data[train_idx], train_data[val_idx]
            train_fold_labels, val_labels = train_labels[train_idx], train_labels[val_idx]
            train_dataset = AugmentedTensorDataset(train_fold_data, train_fold_labels)
            val_dataset = TensorDataset(val_data, val_labels)
            test_dataset = TensorDataset(test_data, test_labels)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            model_params = HYPERPARAMS[model_type]
            model_class = MODEL_REGISTRY[model_type]
            if model_type == 'LSTM':
                model = model_class(
                    input_size=INPUT_SIZE,
                    input_projection_size=model_params['INPUT_PROJECTION_SIZE'],
                    hidden_size=model_params['HIDDEN_SIZE'],
                    dense_size=model_params['DENSE_SIZE'],
                    dropout=model_params['DROPOUT']
                ).to(device)
            else:
                hparams_key = {
                    'CfcClassifier': 'CFC_HPARAMS',
                    'SnnClassifier': 'SNN_HPARAMS',
                    'LsmClassifier': 'LSM_HPARAMS',
                    'EsnClassifier': 'ESN_HPARAMS',
                    'SpikingGNNClassifier': 'SPIKING_GNN_HPARAMS'
                }[model_type]
                model = model_class(
                    input_size=INPUT_SIZE,
                    hidden_size=model_params['HIDDEN_SIZE1'],
                    output_size=1,
                    hparams=model_params[hparams_key],
                    dropout=model_params['DROPOUT'],
                    input_projection_size=model_params['INPUT_PROJECTION_SIZE'],
                    dense_size=model_params['DENSE_SIZE']
                ).to(device)
            shared_hyperparams = {
                'BATCH_SIZE': BATCH_SIZE,
                'EPOCHS': EPOCHS,
                'LEARNING_RATE': LEARNING_RATE,
                'WEIGHT_DECAY': WEIGHT_DECAY,
                'N_FOLDS': N_FOLDS,
                'INPUT_SIZE': INPUT_SIZE,
                'MODEL_TYPE': model_type
            }
            logging.info(f'Fold {fold + 1} Shared Hyperparameters: {shared_hyperparams}')
            logging.info(f'Fold {fold + 1} Model-Specific Hyperparameters ({model_type}): {model_params}')
            num_params = count_parameters(model)
            logging.info(f'Fold {fold + 1} Model ({model_type}) Number of Trainable Parameters: {num_params}')
            optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            criterion = torch.nn.BCELoss()
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            for epoch in tqdm(range(EPOCHS), desc=f"Fold {fold + 1} Epochs ({model_type})"):
                train_loss = train_step(model, train_loader, criterion, optimizer, device)
                train_metrics = evaluate_model(model, train_loader, criterion, device)
                val_metrics = evaluate_model(model, val_loader, criterion, device)
                train_losses.append(train_loss)
                val_losses.append(val_metrics['loss'])
                train_accuracies.append(train_metrics['accuracy'])
                val_accuracies.append(val_metrics['accuracy'])
                scheduler.step(val_metrics['loss'])
                if (epoch + 1) % 10 == 0:
                    logging.info(f'Fold {fold + 1}, Model {model_type}, Epoch {epoch + 1}/{EPOCHS}, '
                                 f'Train Loss: {train_loss:.4f}, '
                                 f'Train Accuracy: {train_metrics["accuracy"]:.4f}, '
                                 f'Val Loss: {val_metrics["loss"]:.4f}, '
                                 f'Val Accuracy: {val_metrics["accuracy"]:.4f}, '
                                 f'Val Sensitivity: {val_metrics["sensitivity"]:.4f}, '
                                 f'Val Specificity: {val_metrics["specificity"]:.4f}, '
                                 f'Val F1: {val_metrics["f1"]:.4f}')
            test_metrics = evaluate_model(model, test_loader, criterion, device)
            logging.info(f'Fold {fold + 1} Test Metrics for {model_type}: '
                         f'Loss: {test_metrics["loss"]:.4f}, '
                         f'Accuracy: {test_metrics["accuracy"]:.4f}, '
                         f'Sensitivity: {test_metrics["sensitivity"]:.4f}, '
                         f'Specificity: {test_metrics["specificity"]:.4f}, '
                         f'F1: {test_metrics["f1"]:.4f}')
            fold_metrics.append({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'test_metrics': test_metrics
            })
        plot_metrics(fold_metrics, output_dir, f'{patient_id}_{model_type}')
        avg_test_metrics = {
            key: np.mean([m['test_metrics'][key] for m in fold_metrics])
            for key in fold_metrics[0]['test_metrics']
        }
        all_model_metrics[model_type] = avg_test_metrics
        logging.info(f'Average Test Metrics for {patient_id} - {model_type}: '
                     f'Loss: {avg_test_metrics["loss"]:.4f}, '
                     f'Accuracy: {avg_test_metrics["accuracy"]:.4f}, '
                     f'Sensitivity: {avg_test_metrics["sensitivity"]:.4f}, '
                     f'Specificity: {avg_test_metrics["specificity"]:.4f}, '
                     f'F1: {avg_test_metrics["f1"]:.4f}')
        logging.info(f'----- Finished training for model: {model_type} -----')
    
    logging.info(f'----- Final Test Metrics Summary for {patient_id} -----')
    for model_type, metrics in all_model_metrics.items():
        logging.info(f'Model {model_type}: '
                     f'Loss: {metrics["loss"]:.4f}, '
                     f'Accuracy: {metrics["accuracy"]:.4f}, '
                     f'Sensitivity: {metrics["sensitivity"]:.4f}, '
                     f'Specificity: {metrics["specificity"]:.4f}, '
                     f'F1: {metrics["f1"]:.4f}')

if __name__ == '__main__':
    main()