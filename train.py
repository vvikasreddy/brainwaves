# from data_preprocess import preprocess_data
from dataset import BrainWaves,BrainWaves1
import torch
from model import Bio, Conv1D_v2, Conv1D_v4
from torch.utils.data import DataLoader
from utils import out_put
import os
from torchnet import  meter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


from matplotlib.lines import Line2D  



def train(modal, dataset, epoch, lr, batch_size, use_gpu, k, exp_name, channels):


    if use_gpu :
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")


    if not os.path.exists(f'./results/{dataset}/{modal}{exp_name}'):
        os.mkdir(f'./results/{dataset}/{modal}{exp_name}')


    file_name = f'./results/{dataset}/{modal}{exp_name}/eeg_val_{k}'

    if dataset == "BrainWaves":
        train_data = BrainWaves1(kind= "train", normalize = 0, channels = channels)
        val_data = BrainWaves1(kind="val", normalize = 0, channels = channels)


    if modal == "transformer":
        model = Bio(input_size=1).to(device)
        print("using transformer model")
    elif modal == "cnn":
        # model = Conv1D_v2(channels = channels).to(device)
        model = Conv1D_v4(8).to(device)
    # shuffle is false, because we are already shuffling the data
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle= True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=True)

    # criterion and optimizer
    criterion = torch.nn.BCELoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    lr = lr

# , weight_decay=1e-5, momentum=0.9
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    # meters
    loss_meter = meter.AverageValueMeter()

    best_accuracy = 0
    best_epoch = 0
    

    # train
    for epoch in range(epoch):
        temp_1 = 0
        temp_0 = 0

        temp_label = 0

        pred_label = []
        true_label = []

        skip = 0
        loss_meter.reset()


        for ii, (data, label) in tqdm(enumerate(train_loader)):

            # print(ii)

            # train model
            if modal == 'faceeeg' or modal == 'faceperi' or modal == 'facebio':
                input = (data[0].float().to(device), data[1].float().to(device))
            else:
                input = data.float().to(device)

            # print(input.shape)
            label = label.float().to(device)

            optimizer.zero_grad()

            pred = model(input).float()
            # print(pred, label)
            pred = pred.view((-1))
            
            temp_1 += torch.count_nonzero(pred)
            temp_0 += (5 - torch.count_nonzero(pred))
            temp_label += torch.count_nonzero(label)
            

            loss = criterion(pred, label)


            loss.backward()

            # Clip gradients
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optimizer.step()

            # meters update
            loss_meter.add(loss.item())

           

            # print(pred, "pred")
            pred = (pred >= 0.5).float().to(device).data
            pred_label.append(pred)
            true_label.append(label)


            # print(label, pred)


        pred_label = torch.cat(pred_label, 0)
        true_label = torch.cat(true_label, 0)


        train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
        out_put('Epoch: ' + 'train' + str(epoch) + '| train accuracy: ' + str(train_accuracy.item()), file_name)

        val_accuracy = val(modal, model, val_loader, use_gpu)

        out_put('Epoch: ' + 'train' + str(epoch) + '| train loss: ' + str(loss_meter.value()[0]) +
                '| val accuracy: ' + str(val_accuracy.item()), file_name)

        print(file_name)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            
            model.save(f"{file_name}_best.pth")

        print(f"epoch {epoch}: successful")
        print( temp_1, "here as there ", temp_0, temp_label)

    model.save(f'{file_name}.pth')

    perf = f"best accuracy is {best_accuracy} in epoch {best_epoch}" + "\n"
    out_put(perf, file_name)

    return best_accuracy


@torch.no_grad()
def val(modal, model, dataloader, use_gpu):
    model.eval()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pred_label = []
    true_label = []

    for ii, (data, label) in enumerate(dataloader):
        if modal == 'faceeeg' or modal == 'faceperi' or modal == 'facebio':
            input = (data[0].float().to(device), data[1].float().to(device))
        else:
            input = data.float().to(device)
        label = label.to(device)
        pred = model(input).float()

        pred = (pred >= 0.5).float().to(device).data

        pred = pred.view(-1)

        pred_label.append(pred)
        true_label.append(label)

    pred_label = torch.cat(pred_label, 0)
    true_label = torch.cat(true_label, 0)

    val_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)

    model.train()

    return val_accuracy