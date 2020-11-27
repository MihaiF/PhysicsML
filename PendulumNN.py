import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

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
    
net = Net(3, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

size_train = 2000
size_valid = 500

def prepare_training_data(set_size):
    x_train = torch.rand(set_size, 3)
    y_train = torch.empty(set_size)
    for i in range(0, set_size):
        #angle = PI * i / set_size
        #x_train[i, 0] = angle / PI
        angle = x_train[i, 0] * PI
        dx = x_train[i, 1]
        dy = x_train[i, 2]
        out = (compute_new_angle_simple(angle, dx, dy) + PI / 2) / PI # normalize in the interval (-pi/2, pi/2)
        y_train[i] = out - x_train[i, 0]
    return x_train, y_train

def prepare_validation_data(set_size):
    x_valid = torch.rand(set_size, 3)
    y_valid = torch.empty(set_size)
    for i in range(0, set_size):
        angle = PI * i / set_size
        x_valid[i, 0] = angle / PI
        x_valid[i, 1] = 0.1
        x_valid[i, 2] = 0.1
        dx = 0.1
        dy = 0.1
        out = (compute_new_angle_simple(angle, dx, dy) + PI / 2) / PI # normalize in the interval (-pi/2, pi/2)
        y_valid[i] = out - x_valid[i, 0]
    return x_valid, y_valid

# prepare training set
x_train, y_train = prepare_training_data(size_train)

# prepare validation set
x_valid, y_valid = prepare_validation_data(size_valid)

# fig, ax = plt.subplots()
# ax.scatter(PI * x_train, PI * y_train)
# fig.savefig("fun.png")

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
torch.save(net.state_dict(), "PendNet.dat")

# validate the NN
y_pred = net(x_valid)
print(y_pred[0], y_pred[10])
print(y_valid[0], y_valid[10])
valid_loss = criterion(y_pred.view(-1), y_valid)
print(valid_loss.item())
    
fig, ax = plt.subplots()
ax.scatter(PI * x_valid[:,0], PI * y_pred.detach().numpy())
ax.scatter(PI * x_valid[:,0], PI * y_valid)
fig.savefig("pred.png")


