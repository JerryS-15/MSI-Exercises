"""
This is the template for coding problems in exercise sheet 5, task 2.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.

Have fun!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

## Define measurements and measurement model
y = np.arange(-3,6) # measurement vector
phi = np.array([[1/6,1/6]]) # regression vector
Phi = np.repeat(phi,9,axis=0) # regressor matrix

# Define the objective function: R^2 -> R
def objectiveFunction(theta: np.ndarray):
    """
    Returns the value of the least-squares objective function for a given theta of shape (2,1).
    """
    residual = y - Phi@theta
    return 0.5*residual.T@residual

### Evaluate the objective function over a grid
theta1_grid, theta2_grid = np.meshgrid(np.linspace(0,5,30),np.linspace(0,5,30))
theta_grid = np.stack([theta1_grid,theta2_grid],axis=2).transpose((2,0,1))
res_grid = y.reshape((9,1,1)) - np.tensordot(Phi,theta_grid,(1,0))
objective_grid = 0.5*np.sum(res_grid*res_grid,axis=0)

### Plot the objective function over the grid
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(theta1_grid,theta2_grid,objective_grid,cmap=cm.jet,alpha=0.5)
ax.set_xlabel("Theta_1")
ax.set_ylabel("Theta_2")
ax.set_zlabel("Objective")

# (c) Find a solution to the ill posed problem using two methods
### YOUR CODE ###
I_Matrix = np.identity(2)
theta_opt_reg = np.dot(np.dot(np.linalg.inv(np.dot(Phi.T, Phi) + 0.2*I_Matrix), Phi.T), y)
u, sigma, vt = np.linalg.svd(Phi)
# S = np.zeros([9, 2])
# for i in range(2):
#     S[i][i] = sigma[i]
# theta_opt_pm = np.dot(np.dot(np.dot(np.dot(vt.T, np.linalg.inv(np.dot(S.T, S) + 0.2*I_Matrix)), S.T), u.T), y)
S0 = np.zeros([9, 2])
for i in range(2):
    S0[i][i] = sigma[i]
    # S0[i][i] = sigma[i]/(sigma[i] * sigma[i] + 0.2)
S_plus = np.linalg.pinv(S0.T@S0)@S0.T
theta_opt_pm = np.dot(np.dot(np.dot(vt.T, S_plus), u.T), y)

#################

# useful function to plot a point on the surface
def plotPoint(x):
    """
    Plots a point with height equal to the objective function defined above on the surface plot.
    Plot a red x. Does nothing if x is None.
    """
    if x is None: return
    ax.plot(x[0],x[1],objectiveFunction(x),'rx')

# plot the points that solve the ill posed least squares problem
plotPoint(theta_opt_reg)
plotPoint(theta_opt_pm)

# show the plot
plt.show()
