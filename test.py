from modules import pruned_linear
import torch
import torch.nn as nn
from utils import assert_equal
import utils

net1 = nn.Linear(5, 2, bias=False)
net1.weight.data.fill_(1)
net2 = nn.Linear(5,2,bias=False)
net2.weight.data.fill_(1)
net2 = pruned_linear(net2, threshold=-1)

optim1 = torch.optim.SGD(net1.parameters(), lr=0.01)
optim2 = torch.optim.SGD(net2.parameters(), lr=0.01)

for i in range(3):
    inp = torch.arange(20).reshape(4,5).float()

    out1 = net1(inp)
    target = torch.randn(out1.shape)
    loss1 = nn.MSELoss()(out1, target)
    loss1.backward()
    optim1.step()

    out2 = net2(inp)
    loss2 = nn.MSELoss()(out2, target)
    loss2.backward()
    optim2.step()

    print(net1.weight)
    print(net2.linear.weight)



