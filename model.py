"""Here we define the fine-tune network that will be appended
to a frozen ResNet152 embedder.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FineTuneNet(nn.Module):
    def __init__(self):
        super(FineTuneNet, self).__init__()
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, 23)	

    def forward(self, x):
        x = F.relu(self.fc1(x))
	x = F.dropout(x, training=self.training)
        x = self.fc2(x)
	return F.log_softmax(x)
