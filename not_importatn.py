import torch
from model import Conv1D_v4,Conv1D_v2, Bio
import numpy as np
from dataset import BrainWavesTest_v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, confusion_matrix, roc_auc_score, f1_score, recall_score, accuracy_score, auc, roc_curve
import matplotlib.pyplot as plt


def load_model(path, model_type, channels, device):
    if model_type == "cnn":
        model = Conv1D_v4(channels).to(device)
    elif model_type == "transformer":
        model = Bio().to(device)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    return model


def plot_performance_metrics(true_labels, ensemble_predictions):
    # Calculate performance metrics
    accuracy = accuracy_score(true_labels, ensemble_predictions)
    precision = precision_score(true_labels, ensemble_predictions)
    recall = recall_score(true_labels, ensemble_predictions)
    f1 = f1_score(true_labels, ensemble_predictions)

    fpr, tpr, _ = roc_curve(true_labels, ensemble_predictions)
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance Metrics', fontsize=18)

    # Bar chart for accuracy, precision, recall, F1-score
    bar_data = [accuracy, precision, recall, f1]
    bar_labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    ax1.bar(bar_labels, bar_data)
    ax1.set_title('Accuracy, Precision, Recall, F1-score', fontsize=14)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # ROC curve
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
def ensemble_predict(models, data):
    predictions = []
    for model in models:
        with torch.no_grad():
            pred = model(data).float()
            pred = (pred >= 0.5).float().cpu().numpy()
        predictions.append(pred)
    
    # Majority voting
    ensemble_pred = np.mean(predictions, axis=0) >= 0.5
    return ensemble_pred.astype(float)

def test_ensemble():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channels = 8
    model_type = "cnn"  # or "transformer"
    
    # Load all models
    models = []
    for k in range(1, 11):
        path = f"D:\\Vikas\\lie_detection\\BrainWaves\\results\\BrainWaves\\cnn_unnormalized_conv4_threshold_0.5_lr_0.0001\\eeg_val_{k}.pth"
        model = load_model(path, model_type, channels, device)
        models.append(model)
    
    # Prepare dataset
    dataset = BrainWavesTest_v2(channels=channels, normalize= 0)
    dataloader = DataLoader(dataset, batch_size=1)
    
    true_labels = []
    ensemble_predictions = []
    
    for data, label, _ in tqdm(dataloader, desc="Processing"):
        input_data = data.float().to(device)
        ensemble_pred = ensemble_predict(models, input_data)
        
        true_labels.append(label.numpy())
        ensemble_predictions.append(ensemble_pred)
    
    true_labels = np.concatenate(true_labels, axis=0)
    ensemble_predictions = np.concatenate(ensemble_predictions, axis=0)
    
    print(true_labels.shape, ensemble_predictions.shape)
    if ensemble_predictions.shape != (2574, 1):
        print("i am here")
        ensemble_predictions = ensemble_predictions.reshape(2574, 1)
    print(true_labels.shape, ensemble_predictions.shape)
    # Calculate metrics
    precision = precision_score(true_labels, ensemble_predictions)
    conf_matrix = confusion_matrix(true_labels, ensemble_predictions)
    roc_auc = roc_auc_score(true_labels, ensemble_predictions)
    f1 = f1_score(true_labels, ensemble_predictions)
    recall = recall_score(true_labels, ensemble_predictions)
    accuracy = accuracy_score(true_labels, ensemble_predictions)
    plot_performance_metrics(true_labels, ensemble_predictions)
    
    print(f'Ensemble Model Results:')
    print(f'Precision: {precision}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'ROC AUC: {roc_auc}')
    print(f'F1 Score: {f1}')
    print(f'Recall: {recall}')
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    test_ensemble()