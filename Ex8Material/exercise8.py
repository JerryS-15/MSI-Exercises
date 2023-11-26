"""
This is the template for coding problems in exercise sheet 8.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.

Have fun!
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from functions import residual, sim_euler

# load data
data = np.load("exercise8_data.npz")  # loading
U = data['u']              # applied controls (dims = 299x2)
xy_meas = data['xy_meas']  # x, y coordinate measurements (dims = 300X2)

# define parameters
N = 300
dT = 0.01
t = dT * np.arange(N)

# initial state
x0 = np.array([0.0, 0.0, 0.0])  

# measurement variances
sigma_meas = np.array([1.6e-3, 4e-4])   # sigma_x^2, sigma_y^2

# include initial guess of the parameters, because we know them from the last exercise
theta_guess = np.array([0.2, 0.2, 0.6])

# 2) Implement the residual function -> see file functions.py
# used to test the function implementation
residual_test = residual(theta_guess, x0, U, t, xy_meas, sigma_meas)

# 3) Compute the solution to the nonlinear least squares problem
############ YOUR CODE ################
# define a vector valued residual function R(theta) that can be used by least_squares
# HINT: create a lambda function
# residual_function = lambda theta: np.sum(xy_meas - sim_euler(t, x0, U, theta)[:,0:1])
def residual_function(theta):
    return residual(theta, x0, U, t, xy_meas, sigma_meas)
# estimate the parameters with least-squares
theta0 = [0.2, 0.2, 0.6]
result = least_squares(residual_function, theta0)
########################################

# extract results
thetaStar = result.x
resnorm = result.cost
res = result.fun
J_thetaStar = result.jac
print(f"thetaStar={thetaStar}")

# 4) Compute the simulated trajectory using $\theta^*$ and use the provided 
# code to plot it versus the measurements and a 4th order polynomial fit.  
############ YOUR CODE ################
# simulate with our estimated parameters
sim_trajectory = sim_euler(t, x0, U, thetaStar)
# simulate with a 4-th poly-fit
order = 4
Phi = np.ones((N, 5))
ThetaLLS = np.zeros((order+1, 1))
for i in range(5):
    for j in range(N):
        Phi[j][i] = t[j]**i
temp_theta = np.dot(np.linalg.inv(np.dot(Phi.T, Phi)), Phi.T)
ThetaLLS = np.dot(temp_theta, xy_meas)
LLS_trajectory = Phi@ThetaLLS
########################################


# Plot the known data
plt.figure(1)
plt.plot(sim_trajectory[:,0], sim_trajectory[:,1], "b")
plt.plot(LLS_trajectory[:,0], LLS_trajectory[:,1], "-.")
plt.plot(xy_meas[:,0], xy_meas[:,1], "rx") 
plt.title("Position $p$")
plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
plt.legend(["Euler Simulated Trajectory","4th Order Polynomial fit", "Measurement Data"])

# Check assumptions on the noise 
trueRes = xy_meas - sim_trajectory[:,0:2]
plt.figure(2)
h2 = plt.hist(trueRes[:,0], 15, edgecolor="black", linewidth=0.5)
plt.title("Histogram of the residuals of the $x$ coordinates")
plt.ylabel("frequency of occurence")
plt.xlabel("$x$ [m]")

plt.figure(3)
h3 = plt.hist(trueRes[:,1], 15, edgecolor="black", linewidth=0.5)
plt.title("Histogram of the residuals of the $y$ coordinates")
plt.ylabel("frequency of occurence")
plt.xlabel("$y$ [m]")


# 6) Approximate the covariance matrix
############ YOUR CODE ################
d = np.size(thetaStar)  # number of estimated parameters
sigma_thetaStar = (np.dot(res.T, res)/(N-d))*np.linalg.inv(np.dot(J_thetaStar.T, J_thetaStar))  # formula on page 48
print(f"sigma_thetaStar={sigma_thetaStar}")
########################################

plt.show()
