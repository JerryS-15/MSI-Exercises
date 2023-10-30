"""
This is the template for coding tasks in exercise sheet 3.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.
=============================================================================================================================

Eventually you will explore the powerful least squares estimation formulated with linear algebra!
And as usual here are two functions that might be helpful to you.

- to stack vectors as columns to form a matrix:
    <https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html>

- to calculate x = A^(-1) b with specifying A, b:
    <https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html>

Have fun!
"""


import matplotlib.pyplot as plt
import numpy as np


## load data
data = np.load("exercise3_task4_dataset.npz")
i1 = data['i1']
u1 = data['u1']
i2 = data['i2']
u2 = data['u2']
N = 2000


## a) Plot each dataset in a corresponding plot
plt.figure(1)
plt.subplot(2,1,1)
# YOUR CODE: plot dataset i1, u1
plt.plot(i1, u1, "x")
plt.subplot(2,1,2)
# YOUR CODE: plot dataset i2, u2
plt.plot(i2, u2, "x")

## b) only PAPER questions :)


## c) 
# Use the least squares estimator to find the experimental values of R and E for each of the two datasets individually.
# YOUR CODE: least squares with dataset i1, u1
# ============================================
# "None" is just to pass the syntax check. Replace them with your code. Don't need to stick to a single line.
# (The same holds for all the following cases)
c1 = np.ones((N))
Phi1 = np.column_stack((c1, i1))
Phi1_plus = np.dot(Phi1.T, Phi1)
theta_star1 = np.dot(np.linalg.pinv(Phi1_plus), Phi1.T)
theta_star1 = np.dot(theta_star1, u1)

# YOUR CODE: least squares with dataset i2, u2
Phi2 = np.column_stack((c1, i2))
Phi2_plus = np.dot(Phi2.T, Phi2)
theta_star2 = np.dot(np.linalg.pinv(Phi2_plus), Phi2.T)
theta_star2 = np.dot(theta_star2, u2)


# Plot the linear fits through the respective measurement data
plt.figure(1)
plt.subplot(2,1,1)
# YOUR CODE: linear fits with dataset i1, u1
x1 = np.linspace(0, 0.01)
y1 = theta_star1[1]*x1 + theta_star1[0]
plt.plot(x1, y1)

plt.subplot(2,1,2)
# YOUR CODE: linear fits with dataset i2, u2
x2 = np.linspace(-0.01, 0.01)
y2 = theta_star2[1]*x2 + theta_star2[0]
plt.plot(x2, y2)


## d) 
# YOUR CODE: calculate the residuals
r1 = np.zeros((N))
r2 = np.zeros((N))

for i in range(0, N):
    r1[i] = theta_star1[1]*i1[i] + theta_star1[0] - u1[i]
    r2[i] = theta_star2[1]*i2[i] + theta_star2[0] - u2[i]

# plot the histograms of them
plt.figure(2)
plt.subplot(1,2,1)
# YOUR CODE
plt.hist(r1)

plt.subplot(1,2,2)
# YOUR CODE
plt.hist(r2)

# show all plots
# ===============
# If you don't see any plots but also no errors, it is very likely because a non-interactive backend is chosen by default.
# One possible solution you can try is to manually select another backend via "matplotlib.use()" function at the beginning.
# Here gives more details: <https://matplotlib.org/stable/users/explain/backends.html>
plt.show()
