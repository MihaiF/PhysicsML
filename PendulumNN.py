import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import random

from torch import nn, optim
from PendulumProj import compute_new_angle_simple

class Net(nn.Module):
  def __init__(self, n_features, n_out):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 80)
    self.fc2 = nn.Linear(80, 80)
    self.fc3 = nn.Linear(80, 80)
    self.fc4 = nn.Linear(80, n_out)
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return torch.sigmoid(self.fc4(x))
    
torch.manual_seed(0)

PI = 3.1415

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(device)
    
net = Net(3, 1)
net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

size_train = 20000
size_valid = 500

def prepare_training_data(set_size):
    x_train = torch.rand(set_size, 3)
    y_train = torch.empty(set_size)
    for i in range(0, set_size):
        angle = x_train[i, 0] * PI
        dx = x_train[i, 1]
        dy = x_train[i, 2]
        out = (compute_new_angle_simple(angle, dx, dy) + PI / 2) / PI # normalize in the interval (-pi/2, pi/2)
        y_train[i] = out - x_train[i, 0]
    return x_train, y_train

def prepare_validation_data(set_size):
    x_valid = torch.rand(set_size, 3)
    y_valid = torch.empty(set_size)
    dx = random.random()
    dy = random.random()
    print(dx, dy)
    for i in range(0, set_size):
        angle = PI * i / set_size
        x_valid[i, 0] = angle / PI #normalize in the interval (0, pi)
        x_valid[i, 1] = dx
        x_valid[i, 2] = dy
        out = (compute_new_angle_simple(angle, dx, dy) + PI / 2) / PI # normalize in the interval (-pi/2, pi/2)
        y_valid[i] = out - x_valid[i, 0]
    return x_valid, y_valid

# prepare training set
x_train, y_train = prepare_training_data(size_train)
x_train = x_train.to(device)
y_train = y_train.to(device)

# prepare validation set
x_valid, y_valid = prepare_validation_data(size_valid)
x_valid = x_valid.to(device)
y_valid = y_valid.to(device)

# fig, ax = plt.subplots()
# ax.scatter(PI * x_train, PI * y_train)
# fig.savefig("fun.png")

train = True
path = "PendNet.dat"
if train:
    # train the NN
    for epoch in range(10000):
        y_pred = net(x_train)
        y_pred = torch.squeeze(y_pred)
        train_loss = criterion(y_pred, torch.squeeze(y_train))
        if epoch % 1000 == 0:
            print(train_loss.item())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
    print(train_loss.item())
    torch.save(net.state_dict(), path)
else:
    net.load_state_dict(torch.load(path))
    
# validate the NN
y_pred = net(x_valid)
print(y_pred[0], y_pred[10])
print(y_valid[0], y_valid[10])
valid_loss = criterion(y_pred.view(-1), y_valid)
print(valid_loss.item())
    
fig, ax = plt.subplots()
x_valid_cpu = x_valid.cpu()
ax.scatter(PI * x_valid_cpu[:,0], PI * y_pred.detach().cpu().numpy())
ax.scatter(PI * x_valid_cpu[:,0], PI * y_valid.cpu())
fig.savefig("pred.png")
# fig.show()
# input()


