import os
import sys
import importlib

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import models
import datasets
from utils.core import pytorch2onnx, rootdir, validate_training_args
from _types import TrainingArgs, BaseDataset

def main(args: TrainingArgs) -> None:

    ### SETUP ###

    # HYPERPARAMS
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    EPOCH = args.epoch
    LR = args.lr
    BATCH_SIZE = args.batchsz

    # LOAD DATASET
    try:
        Dataset = importlib.import_module(f'datasets.{args.dataset}').Dataset
    except ModuleNotFoundError:
        raise ValueError(f'No such dataset in /datasets: {args.dataset}.py')

    root_dir = os.path.join(rootdir, 'data')

    trainset: BaseDataset = Dataset(root_dir, mode='train')
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset: BaseDataset = Dataset(root_dir, mode='test')
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # LOAD MODEL
    try:
        Model: nn.Module = importlib.import_module(f'models.{args.model}').Model
    except ModuleNotFoundError:
        raise ValueError(f'No such model in /models: {args.model}.py')

    model = Model()
    model.train()
    model = model.float().to(device)

    # @TODO: modular criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    # @TODO: allow learning rate scheduling
    optimizer = optim.Adam(model.parameters(), lr=LR)

    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []

    ### TRAINING LOOP ###
    
    print("Start training")
    for epoch in range(EPOCH):  # loop over the dataset multiple times (specify the #epoch)

        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        for j, data in enumerate(trainloader, 0):
            inputs, labels = data['data'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
            accuracy += correct / BATCH_SIZE
            correct = 0.0

            running_loss += loss.item()
            i += 1

        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, i, accuracy / i, running_loss / i))

        Train_loss.append(running_loss / i)
        Train_acc.append((accuracy / i).item())

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0

        for data_test in testloader:
            model.eval()
            inputs_test, labels_test = data_test['data'], data_test['label']
            inputs_test = inputs_test.float().to(device)
            labels_test = labels_test.to(device)
            outputs_test = model(inputs_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted_test == labels_test).sum()

            loss_test = criterion(outputs_test, labels_test)
            running_loss_test += loss_test.item()
            i += 1

        print('Test Acc: %.5f Test Loss: %.5f' % (correct / total, running_loss_test / i))

        Test_loss.append(running_loss_test / i)
        Test_acc.append((correct / total).item())

        # @TODO: save model for each checkpoint
    # @TODO: save final model
    save_path = os.path.join(rootdir, 'checkpoints', args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model, os.path.join(rootdir, 'checkpoints', args.model, 'latest.pkl'))
    pytorch2onnx(model, os.path.join(rootdir, 'checkpoints', args.model, 'latest.onnx'), (1250, 1))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, help='model name')
    argparser.add_argument('--dataset', type=str, help='dataset name')

    argparser.add_argument('--epoch', type=int, help='epoch number', default=2)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--batchsz', type=int, help='batche size', default=32)
    argparser.add_argument('--cuda', type=bool, default=False)

    args = TrainingArgs(**vars(argparser.parse_args()))

    validate_training_args(args)

    try:
        main(args)
    except KeyboardInterrupt:
        pass
        # save_path = os.path.join(rootdir, 'checkpoints', args.model)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # torch.save(model, os.path.join(rootdir, 'checkpoints', args.model, 'latest.pkl'))
