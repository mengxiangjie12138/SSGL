import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import time
from ResNet import ResNet50
import config


def generate_batch(data_loader1, data_loader2):
    source_train_iter1 = iter(data_loader1)
    source_train_iter2 = iter(data_loader2)
    inputs_list = []
    labels_list = []
    inputs, labels = next(source_train_iter1)
    inputs_list.append(inputs)
    for label in labels:
        labels_list.append(label.item())
    inputs, labels = next(source_train_iter2)
    inputs_list.append(inputs)
    for label in labels:
        labels_list.append(label.item())
    inputs = torch.cat(inputs_list, 0)
    labels = torch.tensor(labels_list)
    return inputs, labels


def train(k=6):
    train_dataset_label = dataset.ImageFolder(config.train_dataset_label_path, transform=config.train_transform)
    train_dataloader_label = DataLoader(train_dataset_label, int(config.train_batch_size * 0.2), shuffle=True)
    train_dataset_non_label = dataset.ImageFolder(config.train_dataset_non_label_path, transform=config.train_transform)
    train_dataloader_non_label = DataLoader(train_dataset_non_label, int(config.train_batch_size * 0.8), shuffle=True)
    test_dataset = dataset.ImageFolder(config.test_dataset_path, transform=config.test_transform)
    test_dataloader = DataLoader(test_dataset, int(config.test_batch_size * 0.8), shuffle=False)

    # define GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model
    net = ResNet50(num_classes=config.class_num).to(device)

    cross_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-6, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(config.epoches):
        net.train()
        sum_loss = 0.
        correct = 0.
        correct_non_label = 0.
        correct_label = 0.
        total = 0.
        total_non_label = 0.
        total_label = 0.
        length = config.train_batch_size
        thre = int(config.train_batch_size * 0.2)
        since = time.time()

        for i in range(286):
            inputs, labels = generate_batch(train_dataloader_label, train_dataloader_non_label)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            labels_new = labels[:thre]

            outputs, features, outputs_label, outputs_non_label, theta, adj = net(inputs, k, labels=labels_new)
            loss1 = cross_loss(outputs_label, labels_new)

            loss = loss1

            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            _, pre = torch.max(outputs.data, 1)
            pre_label = pre[:thre]
            pre_non_label = pre[thre:]

            total += labels.size(0)
            correct += torch.sum(pre == labels.data)
            acc = correct.cpu().data.numpy() / total
            total_label += labels_new.size(0)
            correct_label += torch.sum(pre_label == labels_new.data)
            acc_label = correct_label.cpu().data.numpy() / total_label
            total_non_label += labels[thre:].size(0)
            correct_non_label += torch.sum(pre_non_label == labels[thre:].data)
            acc_non_label = correct_non_label.cpu().data.numpy() / total_non_label

            print('[epoch:%d, iter:%d] Loss: %f | Acc: %f | Acc_label: %f | Acc_non_label: %f | Time: %f'
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss/(i + 1), acc, acc_label, acc_non_label, time.time() - since))

        scheduler.step(epoch)

        # start to test
        if epoch % 1 == 0:
            print("start to test:")
            with torch.no_grad():
                correct = 0.
                total = 0.
                loss = 0.

                auxiliary_iter = iter(train_dataloader_label)
                for i, data in enumerate(test_dataloader):
                    net.eval()
                    inputs_test, labels_test = data
                    inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                    inputs_auxiliary, labels_auxiliary = next(auxiliary_iter)
                    inputs_auxiliary, labels_auxiliary = inputs_auxiliary.to(device), labels_auxiliary.to(device)
                    inputs = torch.cat([inputs_auxiliary, inputs_test], 0)
                    outputs, features, outputs_label, outputs_non_label, theta, adj = net(inputs, k, labels=labels_auxiliary)
                    loss += cross_loss(outputs_non_label, labels_test)

                    _, pred = torch.max(outputs_non_label.data, 1)

                    total += labels_test.size(0)
                    correct += torch.sum(pred == labels_test.data)

                test_acc = correct.cpu().data.numpy() / total

                print('the test acc is:{}, the loss is:{}, the time is:{}'.format(test_acc, loss, time.time() - since))


if __name__ == '__main__':
    train()
