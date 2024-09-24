import torch 
from model import Conv1D_v4, Bio
import numpy as np
from dataset import BrainWavesTest_v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score

# enter the model_name
what = "cnn"

def test():
    with torch.no_grad():
        pred_label = []
        true_label = []

        for ii, (data, label, path) in enumerate(dataloader):
            
            input = data.float().to(device)
            label = label.to(device)
            pred = model(input).float()

            pred = (pred >= 0.58).float().to(device).data

            pred = pred.view(-1)

            pred = pred.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            pred_label.append(pred)
            true_label.append(label)

            # this piece of snippet is to get number of correct labels per subject
            # print(path)
            subject = path[0].split('_')[-2]
            p = 0 if pred <0.5 else 1
            if accu.get(subject, -1) == -1:
                ans = 0
                if p == label:
                    ans = 1
                accu[subject] = [ans, 1]
            else:
                if p == label:
                    accu[subject][0] += 1
                    accu[subject][1] += 1
                else:
                    accu[subject][1] += 1

        


        pred_label = np.concatenate(pred_label, axis=0)

        true_label = np.concatenate(true_label, axis=0)

        # val_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)

        # print(val_accuracy)
        precision = precision_score(true_label, pred_label)
        print(f'Precision: {precision}')

        from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, accuracy_score

        conf_matrix = confusion_matrix(true_label, pred_label)
        print(f'Confusion Matrix:\n{conf_matrix}')

        roc_auc = roc_auc_score(true_label, pred_label)
        print(f'ROC AUC: {roc_auc}')
        
        f1 = f1_score(true_label, pred_label)
        print(f'f1_score: {f1}')

        recall = recall_score(true_label, pred_label)
        print(f"recall", recall)

        print("accuracy", accuracy_score(true_label, pred_label))

        return accu
        
for k in range(1,11):
    channels = 8
    dataset = BrainWavesTest_v2(channels = channels, normalize= 0)

    dataloader = DataLoader(dataset, batch_size = 1)
    
    print(k, "-----------------------------")
    path = f"D:\\Vikas\\lie_detection\\BrainWaves\\results\\BrainWaves\\cnn_unnormalized_conv4_threshold_0.5_lr_0.0001\\eeg_val_{k}.pth"
    device = "cuda"
    channels = 8
    if what == "cnn":
        model = Conv1D_v4(channels).to(device)
    if what == "transformer":
        model = Bio().to(device)   


    state_dict= torch.load(path)
    model.load_state_dict(state_dict= state_dict)
    model.eval()

    accu = {}

    test()