import math
import numpy as np
import scipy.optimize as opt

l = 1 # pendulum (rest) length

theta = 0.1 # input angle
dx = 0.2 # input x displacement
dy = 0.3 # input y displacement

def compute_new_angle_simple(theta, dx, dy):
    x = l * math.cos(theta)
    y = l * math.sin(theta)
    x0 = np.array([x + dx, y + dy])
    #print(x0)
    len = math.sqrt(x0[0] * x0[0] + x0[1] * x0[1])
    x1 = x0 * l / len
    #print(x1)
    theta1 = math.atan2(x1[1], x1[0])
    return theta1

def compute_new_angle_scipy(theta, dx, dy):
    # compute the position
    x = l * math.cos(theta)
    y = l * math.sin(theta)
    #print(x, y)

    # displace the position
    x0 = np.array([x + dx, y + dy])
    print(x0)

    def proj_objective(x):
        return (x[0] - x0[0]) ** 2  + (x[1] - x0[1]) ** 2
        
    def proj_constraint(x):
        return math.sqrt(x[0] * x[0] + x[1] * x[1]) - l;
        
    # solve the projection minimization   
    con = {'type' : 'eq', 'fun' : proj_constraint }
    sol = opt.minimize(proj_objective, x0, constraints=con)
    print(sol.x)

    x1 = sol.x
    #print(proj_constraint(x1))
    theta1 = math.atan2(x1[1], x1[0])
    #print(theta1)
    return theta1
    
#theta_new = compute_new_angle_simple(theta, dx, dy)
#print(theta_new)

#theta_new = compute_new_angle_scipy(theta, dx, dy)
#print(theta_new)