import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import plotext as plot

torch.manual_seed(2)

# convert tensor to image
def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.numpy().squeeze()
  return image

# the neural network creation
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.feature_mask2 = nn.Parameter(torch.ones((28, 28)), requires_grad=False)
    self.feature_mask = nn.Parameter(torch.ones(28, 28))
    # a convolutional layer with 1 input channel (grayscale), 10 output channels, a kernel size of 5, and a stride of 1
    self.conv1 = nn.Conv2d(1,10,kernel_size=5,stride=1)
    self.conv2 = nn.Conv2d(10,10,kernel_size=5,stride=1)
    # a maxpool layer with a kernel size of 2 and a stride of 2
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2) #2x2 maxpool
    # a fully connected layer with 100 output features
    self.fc1 = nn.Linear(4*4*10,100)
    # a fully connected layer with 10 output features - the 10 possible digit classes
    self.fc2 = nn.Linear(100,10)

  def forward(self,x):
    # mask input features
    x = x * self.feature_mask[None, None, :, :]
    x = x * self.feature_mask2[None, None, :, :]

    x = F.relu(self.conv1(x)) #24x24x10
    x = self.pool(x) #12x12x10
    x = F.relu(self.conv2(x)) #8x8x10
    x = self.pool(x) #4x4x10
    x = x.view(-1, 4*4*10) #flattening
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# load the data
# this is also a torch.utils.data.Dataset
train_ds = datasets.MNIST('../data',train=True,download=True, transform=transforms.Compose([transforms.ToTensor()]))
batch_size = 100
validation_split = .1
shuffle_dataset = True
random_seed= 2
l1_lambda = 0.01
l1_lambda_end = 0.01
iterations = 10
max_pruned = 0.8

# fn from https://arxiv.org/pdf/1710.01878.pdf
# https://www.wolframalpha.com/input?i=plot+y+%3D+10+-+10+*+%281+-+x%29%5E3+from+x+%3D+0+to+1
def prune_percent(f):
    # f is fraction of run completed between 0 and 1
    if f < 0.5:
        return 0
    else:
        f2 = (f - 0.5)/0.5
        return max_pruned - max_pruned * (1 - f2) ** 3

# Creating data indices for training and validation splits:
dataset_size = len(train_ds)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)


# pytorch DataLoaders
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                                sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=False,download=True,
      transform=transforms.Compose([transforms.ToTensor()])),batch_size=batch_size,shuffle=True)

l1_batch_delta = (l1_lambda_end - l1_lambda) / (iterations * len(train_loader))
# fig, ax = plt.subplots(nrows=2,ncols=3)

# i=0
# for row in ax:
#   for col in row:
#     col.imshow(im_convert(train_loader.dataset[i][0]))
#     col.set_title("digit "+str(train_loader.dataset[i][1]))
#     col.axis("off")
#     i+=1

# plt.show()

model = Net().cuda()
optimizer = optim.SGD(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

train_errors = []
train_acc = []
val_errors = []
val_acc = []
n_train = len(train_loader)*batch_size
n_val = len(validation_loader)*batch_size

print('training..')
percent_through = 0
for i in range(iterations):
  total_loss = 0
  total_acc = 0
  c = 0
  for (images,labels) in train_loader:
    l1_lambda += l1_batch_delta
    images = images.cuda()
    labels = labels.cuda()

    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output,labels)
    l1_reg = torch.abs(model.feature_mask).sum()
    loss = loss + l1_lambda * l1_reg
    loss.backward()
    optimizer.step()

    percent_through += 1/(iterations*len(train_loader))
    p = prune_percent(percent_through)
    if p > 0:
        # set the lowest p fraction to zero
        k = int(p * model.feature_mask.numel())
        if k > 0:
            kth_smallest_value = model.feature_mask.flatten().kthvalue(k)[0]
            # prune the smallest weights
            model.feature_mask2[model.feature_mask < kth_smallest_value] = 0

    total_loss+=loss.item()
    total_acc+=torch.sum(torch.max(output,dim=1)[1]==labels).item()*1.0
    c+=1

  #validation
  total_loss_val = 0
  total_acc_val = 0
  c = 0
  for images,labels in validation_loader:
    images = images.cuda()
    labels = labels.cuda()
    output = model(images)
    loss = criterion(output,labels)

    total_loss_val +=loss.item()
    total_acc_val +=torch.sum(torch.max(output,dim=1)[1]==labels).item()*1.0
    c+=1

  num_masked = (model.feature_mask2 == 0).sum().item() / model.feature_mask2.numel()
  print('iteration ',i,'train loss: ',total_loss/n_train,'train acc: ',total_acc/n_train, 'val loss: ',total_loss_val/n_val,'val acc: ',total_acc_val/n_val, 'masked: ', num_masked, 'l1_lambda: ', l1_lambda, 'p: ', p)
  # plot.hist(list(model.feature_mask.flatten()) + [0,1], bins=10)
  # plot.plot_size(100, 10)
  # plot.show()
  # plot.clear_figure()
  if num_masked > 0:
      plt.imshow(model.feature_mask2.cpu().detach().numpy(), cmap='Greys')
      plt.axis('off')
      plt.show()
      plt.savefig(f'/home/sca63/temp/iter{i}_mask.png', bbox_inches='tight', pad_inches=0, dpi=100)
      # show the image in the command line
      plot.image_plot(f'/home/sca63/temp/iter{i}_mask.png')
      plot.plot_size(60, 20)
      plot.show()
      plot.clear_figure()


  train_errors.append(total_loss/n_train)
  train_acc.append(total_acc/n_train)
  val_errors.append(total_loss_val/n_val)
  val_acc.append(total_acc_val/n_val)

# total_acc = 0
# for images,labels in test_loader:
  # images = images.cuda()
  # labels = labels.cuda()
  # output = model(images)
  # total_acc+=torch.sum(torch.max(output,dim=1)[1]==labels).item()*1.0

# print("Test accuracy :",total_acc/len(test_loader.dataset))
