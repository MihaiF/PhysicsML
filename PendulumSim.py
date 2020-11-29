import math
import matplotlib.pyplot as plt
import numpy as np
import torch

from PendulumProj import compute_new_angle_simple
from PendulumNN import net

def project(x, y):
    len = math.sqrt(x * x + y * y)
    x1 = x * l / len
    y1 = y * l / len
    return x1, y1

# simple integration using PBD

l = 1 # pendulum (rest) length
PI = 3.1415

# initial conditions
theta0 = 0.0
x0 = l * math.cos(theta0)
y0 = l * math.sin(theta0)
theta0dot = 0.2
x0dot = -l * theta0dot * math.sin(theta0)
y0dot = l * theta0dot * math.cos(theta0)

dt = 1 # time step in seconds
steps = 20
x = np.empty(steps)
y = np.empty(steps)
x[0] = x0;
y[0] = y0
xdot = np.empty(steps)
ydot = np.empty(steps)
xdot[0] = x0dot;
ydot[0] = y0dot
for i in range(0, steps - 1):
    x1 = x[i] + dt * xdot[i]
    y1 = y[i] + dt * ydot[i]
    theta0 = math.atan2(y[i], x[i])    
    theta1 = compute_new_angle_simple(theta0, dt * xdot[i], dt * ydot[i])
    args = torch.tensor([theta0 / PI, dt * xdot[i], dt * ydot[i]], dtype=torch.float)
    out = net(args)
    theta1net = theta0 + PI * (out - 0.5)
    print(theta1, theta1net.item())    
    theta1 = theta1net;
    x[i+1] = l * math.cos(theta1)
    y[i+1] = l * math.sin(theta1)
    # x2, y2 = project(x1, y1)
    # x[i+1] = x2
    # y[i+1] = y2
    xdot[i+1] = (x[i+1] - x[i]) / dt
    ydot[i+1] = (y[i+1] - y[i]) / dt

fig, ax = plt.subplots()
plt.axis('square')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.scatter(x, y)
fig.savefig("sim.png")
fig.show()
input()
    