"""
This is the template for coding tasks in exercise sheet 4.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.
=============================================================================================================================

We are having a bit more exploration with the Least Squares problem, but this time with weighting!


New functions that can be useful for this exercise:
- to generate a diagonal matrix by specifying its diagonal elements as an array: 
    <https://numpy.org/doc/stable/reference/generated/numpy.diag.html>

- to calculate the eigenvalues and eigenvectors of a matrix:
    <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html>


Recall some useful functions that already showed up in previous exercises:
- np.column_stack(): combine vectors as column entries into a matrix
- np.linspace() / np.arange(): generate a series
- np.linalg.solve(): solve x for Ax = b
- np.mean(): calculate the mean value, remember to specify the "axis" parameter properly.
- plt.plot(): plot data as a line / scatter / etc.


Have fun!
"""


import matplotlib.pyplot as plt
import numpy as np


## load the data
data = np.load("exercise4_dataset.npz")

I = data['I']        # voltage data
U = data['U']        # current data
N_e = np.size(I, 0)  # number of students/experiments
N_m = np.size(I, 1)  # number of measurements per experiment


## (a) Plot all measurements 
plt.figure(1)
for d in range(N_e):
    plt.plot(I[d,:], U[d,:], "x")
plt.xlabel(r"$I$")
plt.ylabel(r"$U$")


## (b) LLS/WLS for student 1 and plot
# YOUR CODE: compute LLS/WLS estimation and fit
c1 = np.ones((N_m))
Phi = np.column_stack((I[0,:], c1))

theta_LLS_1 = np.dot(np.linalg.pinv(np.dot(Phi.T, Phi)), Phi.T)
theta_LLS_1 = np.dot(theta_LLS_1, U[0,:])
U_LLS_1 = np.zeros((N_m))
for j in range((N_m)):
    U_LLS_1[j] = theta_LLS_1[0]*I[0,j] + theta_LLS_1[1]

W_0 = np.zeros((N_m))
for j in range(N_m):
    W_0[j] = 1 / (j + 1)

W = np.diag(W_0)

theta_WLS_1 = np.dot(np.dot(Phi.T, W), Phi)
theta_WLS_1 = np.dot(np.linalg.pinv(theta_WLS_1), Phi.T)
theta_WLS_1 = np.dot(np.dot(theta_WLS_1, W), U[0,:])
U_WLS_1 = np.zeros((N_m))
for j in range((N_m)):
    U_WLS_1[j] = theta_WLS_1[0]*I[0,j] + theta_WLS_1[1]

# Plotting
plt.figure(2)
# YOUR CODE: plot in the order of data, LLS fit, WLS fit  of student 1
# for j in range(N_m):
plt.plot(I[0,:], U[0,:], 'x')
plt.plot(I[0,:], U_LLS_1)
plt.plot(I[0,:], U_WLS_1)

# Format the plot
plt.legend(["data", "LLS", "WLS"], loc="upper left")
plt.xlabel(r"$I$")
plt.ylabel(r"$U$")


## (c) LLS/WLS for all students
thetas_LLS = np.zeros((N_e,2))
thetas_WLS = np.zeros((N_e,2))
for d in range(N_e):
    # YOUR CODE: compute LLS/WLS estimation of each experiment
    Phi = np.column_stack((I[d,:], c1))
    thetas_LLS[d,:] = np.dot(np.dot(np.linalg.pinv(np.dot(Phi.T, Phi)), Phi.T), U[d,:])
    thetas_WLS[d,:] = np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(np.dot(Phi.T, W), Phi)), Phi.T), W), U[d,:])
    

## (d) Estimate the mean and covariance of theta
# YOUR CODE: mean & covariance of LLS
theta_mean_LLS = np.sum(thetas_LLS, axis=0)/N_e
thetas_LLS_centered = np.zeros((2, 2))
for j in range(N_e):
    tempMatrix = thetas_LLS[j,:] - theta_mean_LLS
    tempMatrix1 = [[tempMatrix[0]*tempMatrix[0], tempMatrix[1]*tempMatrix[0]],[tempMatrix[0]*tempMatrix[1], tempMatrix[1]*tempMatrix[1]]]
    thetas_LLS_centered = thetas_LLS_centered + tempMatrix1
sigma_LLS = thetas_LLS_centered/(N_e - 1)

# YOUR CODE: mean & covariance of WLS
theta_mean_WLS = np.sum(thetas_WLS, axis=0)/N_e
thetas_WLS_centered = np.zeros((2, 2))
tempMatrix2 = np.zeros((2, 2))
for j in range(N_e):
    tempMatrix = thetas_WLS[j,:] - theta_mean_WLS
    tempMatrix2 = [[tempMatrix[0]*tempMatrix[0], tempMatrix[1]*tempMatrix[0]],[tempMatrix[0]*tempMatrix[1], tempMatrix[1]*tempMatrix[1]]]
    thetas_WLS_centered = thetas_WLS_centered + tempMatrix2
sigma_WLS = thetas_WLS_centered/(N_e - 1)


## (e) Plot for all students
plt.figure(3)

# Plot all estimation
plt.plot(thetas_LLS[:,0], thetas_LLS[:,1], "rx")
plt.plot(thetas_WLS[:,0], thetas_WLS[:,1], "bx")

# Plot the mean for estimators 
plt.plot(theta_mean_LLS[0], theta_mean_LLS[1], "r.", markersize=10)
plt.plot(theta_mean_WLS[0], theta_mean_WLS[1], "b.", markersize=10)

# Compute the eigenvalues and eigenvectors for estimators
# YOUR CODE: properly call the function np.linalg.eig()
[w_LLS, V_LLS] = np.linalg.eig(np.cov(thetas_LLS.T))
[w_WLS, V_WLS] = np.linalg.eig(np.cov(thetas_WLS.T))

# Generate coordinates as 50 points on a unit circle
num_xy = 50
xy = np.vstack((
    np.cos(np.linspace(0, 2*np.pi, num_xy)),
    np.sin(np.linspace(0, 2*np.pi, num_xy)),
))

# Generate the points of the confidence ellipse
xy_ellipse1 = np.outer(theta_mean_LLS, np.ones(num_xy)) + V_LLS@np.sqrt(np.diag(w_LLS))@xy
xy_ellipse2 = np.outer(theta_mean_WLS, np.ones(num_xy)) + V_WLS@np.sqrt(np.diag(w_WLS))@xy

# Plot the confidence ellipse
plt.plot(xy_ellipse1[0,:], xy_ellipse1[1,:], "r-")
plt.plot(xy_ellipse2[0,:], xy_ellipse2[1,:], "b-")

# Format the plot
plt.legend([
    r"$\theta_{LLS}^{(d)}$", r"$\theta_{WLS}^{(d)}$",
    r"$\bar{\theta}_{LLS}$", r"$\bar{\theta}_{WLS}$",
    r"$\Sigma_{LLS}$", r"$\Sigma_{WLS}$",
])
plt.xlabel(r"$R_0^*$")
plt.ylabel(r"$E_0^*$")


## Show all plots
# ===============
# If you don't see any plots but also no errors, it is very likely because a non-interactive backend is chosen by default.
# One possible solution you can try is to manually select another backend via "matplotlib.use()" function at the beginning.
# Here gives more details: <https://matplotlib.org/stable/users/explain/backends.html>
plt.show()
