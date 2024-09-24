
from train import train
import argparse
import os


def demo():
    parser = argparse.ArgumentParser(description='Per-subject experiment')
    parser.add_argument('--dataset', '-d', default='BrainWaves', help='The dataset used for evaluation', type=str)
    parser.add_argument('--epoch', '-e', default=150, help='The number of epochs in training', type=int)
    parser.add_argument('--batch_size', '-b', default=64, help='The batch size used in training', type=int)
    parser.add_argument('--learn_rate', '-l', default=0.0001, help='Learn rate in training', type=float)
    parser.add_argument('--gpu', '-g', default='True', help='Use gpu or not', type=str)
    parser.add_argument('--modal', '-m', default='cnn', help='Type of data to train', type=str)

    args = parser.parse_args()

    use_gpu = True if args.gpu == 'True' else False
    exp_name = "_unnormalized_conv4_threshold_0.5_lr_0.0001"
    if not os.path.exists(f'./results/'):
        os.mkdir(f'./results/')
    if not os.path.exists(f'./results/{args.dataset}/'):
        os.mkdir(f'./results/{args.dataset}/')
    if not os.path.exists(f'./results/{args.dataset}/{args.modal}{exp_name}/'):
        os.mkdir(f'./results/{args.dataset}/{args.modal}{exp_name}/')

    for k in range(1,11):
        train(modal=args.modal,dataset=args.dataset,epoch=args.epoch, lr = args.learn_rate, batch_size=args.batch_size, use_gpu = use_gpu, k = k
        , exp_name = exp_name, channels = 8)

demo()