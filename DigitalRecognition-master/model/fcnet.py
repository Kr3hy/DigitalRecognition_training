import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from config.global_config import global_config
from data.dataset import NumberDataset
from data.classes import NUMBER_CLASSES
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.resize=torch.Reshape()
            # transforms.Resize([1,560])  #column , row
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=20*28,out_features=256),nn.ReLU()) # 1680# 560#??? not 560???  that s causes by channels problem
        self.layer2 = nn.Sequential(nn.Linear(in_features=256, out_features=84), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(in_features=84,out_features=6)
                                    # ,nn.Softmax(dim=0)
                                    )  # 最后一层接Softmax所以不需要ReLU激活
        # self.layer4=nn.Sequential(nn.Linear(84,5),nn.Softmax())

    def forward(self, x):
        x=x.reshape(-1,560)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x)
        return x