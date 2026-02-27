import os
import sys
import time
import random
import pickle
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    # Please install apex for mixed precision training from: https://github.com/NVIDIA/apex
    apex_support = False

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNetNew(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        f = self.avgpool(c5)

        return c2, c3, c4, c5, f


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetNew(block, layers, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[arch]))
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


class FpnResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super(FpnResNet18, self).__init__()

        self.R18 = resnet18(pretrained)

        # Top layer
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps."""

        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c2, c3, c4, c5, f = self.R18(x)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5, f


class FpnResNetANAUS(nn.Module):
    """ The ResNet feature extractor + projection head + classifier for ANAUS """

    def __init__(self, out_dim, num_classes, pretrained=False):
        super(FpnResNetANAUS, self).__init__()

        # self.features = nn.Sequential(*list(resnet.children())[:-1])  # discard the last fc layer
        self.features = FpnResNet18(pretrained=True)  # default=True

        self.l2 = nn.Linear(256 * 56 * 56, out_dim)
        self.l3 = nn.Linear(256 * 28 * 28, out_dim)
        self.l4 = nn.Linear(256 * 14 * 14, out_dim)
        self.l5 = nn.Linear(256 * 7 * 7, out_dim)
        self.lf = nn.Linear(512 * 1 * 1, out_dim)

        # projection MLP
        self.linear = nn.Linear(5 * out_dim, out_dim)

        # classifier
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        p2, p3, p4, p5, h = self.features(x)
        f2, f3, f4, f5, fp = (torch.flatten(p2, start_dim=1), torch.flatten(p3, start_dim=1),
                              torch.flatten(p4, start_dim=1), torch.flatten(p5, start_dim=1),
                              torch.flatten(h, start_dim=1))

        x2 = F.relu(self.l2(f2))
        x3 = F.relu(self.l3(f3))
        x4 = F.relu(self.l4(f4))
        x5 = F.relu(self.l5(f5))
        xf = F.relu(self.lf(fp))

        x = torch.cat((x2, x3, x4, x5, xf), 1)
        x = self.linear(x)
        x = self.fc(x)

        return x


class COVIDDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        """
        POCUS Dataset
            param data_dir: str
            param transform: torch.transform
        """
        self.label_name = {"covid19": 0, "pneumonia": 1, "regular": 2}
        with open(data_dir, 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
        if train:
            self.X, self.y = X_train, y_train  # [N, C, H, W], [N]
        else:
            self.X, self.y = X_test, y_test  # [N, C, H, W], [N]
        self.transform = transform

    def __getitem__(self, index):
        img_arr = self.X[index].transpose(1, 2, 0)  # CHW => HWC
        img = Image.fromarray(img_arr.astype('uint8')).convert('RGB')  # 0~255
        label = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.y)

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args):
    # ============================ step 1/5 data ============================
    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])

    # MyDataset
    train_data = COVIDDataset(data_dir=args.data_path, train=True, transform=train_transform)
    valid_data = COVIDDataset(data_dir=args.data_path, train=False, transform=valid_transform)

    # DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size)

    # ============================ step 2/5 model ============================
    net = FpnResNetANAUS(out_dim=256, num_classes=3, pretrained=True)

    if os.path.exists(args.ckpt_path):  # import pretrained model weights
        state_dict = torch.load(args.ckpt_path)
        new_dict = {f'features.R18.{k}': v for k, v in state_dict.items()}  # Discard MLP and fc
        model_dict = net.state_dict()
        model_dict.update(new_dict)
        net.load_state_dict(model_dict)
        print('\nThe self-supervised trained parameters are loaded.\n')
    else:
        print('\nThe self-supervised trained parameters are not loaded.\n')

    # Global-local (3 layers: last stage (c5) and toplayer (p5))
    for name, param in net.named_parameters():
        if (not name.startswith('features.R18.layer4.1')) and (not name.startswith('features.toplayer')):
            param.requires_grad = False

    for name, param in net.named_parameters():
        print(name, '\t', 'requires_grad=', param.requires_grad)

    device = torch.device(args.device)
    net.to(device)

    # ============================ step 3/5 loss function ============================
    criterion = nn.CrossEntropyLoss()  # choose loss function

    # ============================ step 4/5 optimizer ================================
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # choose optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0,
                                                     last_epoch=-1)  # set learning rate decay strategy

    # ============================ step 5/5 train and evaluate ============================
    print('\nTraining start!\n')
    start = time.time()
    train_curve = list()
    valid_curve = list()
    max_acc = 0.
    reached = 0  # which epoch reached the max accuracy

    # the statistics of classification result: classification_results[true][pred]
    classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    best_classification_results = None

    if apex_support and args.fp16:
        net, optimizer = amp.initialize(net, optimizer,
                                        opt_level='O2',
                                        keep_batchnorm_fp32=True)

    for epoch in range(args.epochs):

        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, data in enumerate(train_loader):
            # forward
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels.long())

            if apex_support and args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # update weights
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()

            # print training information
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % args.log_interval == 0:
                loss_mean = loss_mean / args.log_interval
                print("\nTraining:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, args.epochs, i + 1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        # print('Learning rate this epoch:', scheduler.get_last_lr()[0])  # python >=3.7
        print('Learning rate this epoch:', scheduler.base_lrs[0])  # python 3.6
        scheduler.step()  # updata learning rate

        # validate the model
        if (epoch + 1) % args.val_interval == 0:
            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).cpu().squeeze().sum().numpy()
                    for k in range(len(predicted)):
                        classification_results[labels[k]][predicted[k]] += 1  # "label" is regarded as "predicted"

                    loss_val += loss.item()

                acc = correct_val / total_val

                if acc > max_acc:  # record best accuracy
                    max_acc = acc
                    reached = epoch
                    best_classification_results = classification_results
                classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                valid_curve.append(loss_val / valid_loader.__len__())
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, args.epochs, j + 1, len(valid_loader), loss_val, acc))

    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(args.epochs, round(time.time() - start)))
    print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc, reached))

    print('\nThe best prediction results of the dataset:')
    print('Class 0 predicted as class 0:', best_classification_results[0][0])
    print('Class 0 predicted as class 1:', best_classification_results[0][1])
    print('Class 0 predicted as class 2:', best_classification_results[0][2])
    print('Class 1 predicted as class 0:', best_classification_results[1][0])
    print('Class 1 predicted as class 1:', best_classification_results[1][1])
    print('Class 1 predicted as class 2:', best_classification_results[1][2])
    print('Class 2 predicted as class 0:', best_classification_results[2][0])
    print('Class 2 predicted as class 1:', best_classification_results[2][1])
    print('Class 2 predicted as class 2:', best_classification_results[2][2])

    acc0 = best_classification_results[0][0] / sum(best_classification_results[i][0] for i in range(3))
    recall0 = best_classification_results[0][0] / sum(best_classification_results[0])
    print('\nClass 0 accuracy:', acc0)
    print('Class 0 recall:', recall0)
    print('Class 0 F1:', 2 * acc0 * recall0 / (acc0 + recall0))

    acc1 = best_classification_results[1][1] / sum(best_classification_results[i][1] for i in range(3))
    recall1 = best_classification_results[1][1] / sum(best_classification_results[1])
    print('\nClass 1 accuracy:', acc1)
    print('Class 1 recall:', recall1)
    print('Class 1 F1:', 2 * acc1 * recall1 / (acc1 + recall1))

    acc2 = best_classification_results[2][2] / sum(best_classification_results[i][2] for i in range(3))
    recall2 = best_classification_results[2][2] / sum(best_classification_results[2])
    print('\nClass 2 accuracy:', acc2)
    print('Class 2 recall:', recall2)
    print('Class 2 F1:', 2 * acc2 * recall2 / (acc2 + recall2))

    return best_classification_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--ckpt_path', required=True, help='folder of ckpt')
    parser.add_argument("--data_dir", required=True, help="data dir to pickle file")
    parser.add_argument("--epochs", default=30, type=int, help='epochs')
    parser.add_argument("--batch_size", default=32, type=int, help='batch size')
    parser.add_argument("--lr", default=0.01, type=float, help='learning rate')
    parser.add_argument("--weight_decay", default=1e-4, type=float, help='weight decay')
    parser.add_argument("--fp16", default=False, action="store_true", help='fp16')
    parser.add_argument("--device", default='cuda', type=str, help='device')
    parser.add_argument("--log_interval", default=10, type=int, help='log interval')
    parser.add_argument("--val_interval", default=1, type=int, help='val interval')
    args = parser.parse_args()

    set_seed(1)  # random seed

    result_file = './result.txt'

    if (not (os.path.exists(result_file))) and (os.path.exists(args.ckpt_path)):
        confusion_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for i in range(0, 5):
            print('\n' + '=' * 20 + 'The training of fold {} start.'.format(i+1) + '=' * 20)
            args.data_path = os.path.join(args.data_dir, "covid_data{}.pkl".format(i+1)) # Pocus Dataset

            best_classification_results = main(args=args)
            confusion_matrix = confusion_matrix + np.array(best_classification_results)

        print('\nThe confusion matrix is:')
        print(confusion_matrix)
        print('\nThe precision of class 0 is:', confusion_matrix[0, 0] / sum(confusion_matrix[:, 0]))
        print('The precision of class 1 is:', confusion_matrix[1, 1] / sum(confusion_matrix[:, 1]))
        print('The precision of class 2 is:', confusion_matrix[2, 2] / sum(confusion_matrix[:, 2]))
        print('\nThe recall of class 0 is:', confusion_matrix[0, 0] / sum(confusion_matrix[0]))
        print('The recall of class 1 is:', confusion_matrix[1, 1] / sum(confusion_matrix[1]))
        print('The recall of class 2 is:', confusion_matrix[2, 2] / sum(confusion_matrix[2]))
        print('\nTotal acc is:',
              (confusion_matrix[0, 0] + confusion_matrix[1, 1] + confusion_matrix[2, 2]) / confusion_matrix.sum())

        file_handle = open(result_file, mode='w+')
        file_handle.write("precision 0: " + str(confusion_matrix[0, 0] / sum(confusion_matrix[:, 0])))
        file_handle.write('\r\n')
        file_handle.write("precision 1: " + str(confusion_matrix[1, 1] / sum(confusion_matrix[:, 1])))
        file_handle.write('\r\n')
        file_handle.write("precision 2: " + str(confusion_matrix[2, 2] / sum(confusion_matrix[:, 2])))
        file_handle.write('\r\n')
        file_handle.write("recall 0: " + str(confusion_matrix[0, 0] / sum(confusion_matrix[0])))
        file_handle.write('\r\n')
        file_handle.write("recall 1: " + str(confusion_matrix[1, 1] / sum(confusion_matrix[1])))
        file_handle.write('\r\n')
        file_handle.write("recall 2: " + str(confusion_matrix[2, 2] / sum(confusion_matrix[2])))
        file_handle.write('\r\n')
        file_handle.write("Total acc: " + str(
            (confusion_matrix[0, 0] + confusion_matrix[1, 1] + confusion_matrix[2, 2]) / confusion_matrix.sum()))

        file_handle.close()
