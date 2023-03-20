import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm

class fullConnection(torch.nn.Module):
    def __init__(self,classes):
        super(fullConnection, self).__init__()
        self.phase_sequential = nn.Sequential()
        self.classes=classes

        # self.phase_sequential.add_module(nn.Linear(560,256)) #256channels : 256 features ; like matrix , the first param is row
        # self.phase_sequential.add_module(nn.ReLU)
        # self.phase_sequential.add_module(nn.Linear(256,84))
        # self.phase_sequential.add_module(nn.ReLU)
        # self.phase_sequential.add_module(nn.Linear(84,classes))
        # self.phase_sequential.add_module(nn.Softmax)


    def forward(self, inputs):
        model=nn.Sequential(torch.reshape(inputs,(-1, 1)),
                            nn.Linear(560, 256),
                            nn.ReLU(),
                            nn.Linear(256, 84),
                            nn.ReLU(),
                            nn.Linear(84, self.classes),
                            nn.Softmax
                            )
        out=self.model(inputs)
        return out




