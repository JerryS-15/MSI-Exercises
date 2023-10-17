"""
This is the template for coding tasks in exercise sheet 1.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.
=============================================================================================================================

Through this exercise, you will get to know the basic usage of NumPy and Matplotlib, i.e. frequently used functions/methods.
Because it is the starting point, we thought some referrences to the official documentation will ease the process.

PRECAUTION: you don't need to read through every details in the documentation, that is way too overkill. In most cases, 
grasp a rough idea of how to call a function/method with minimal arguments, and you are good to start writing your own code! 
Many typically offer some examples, which are even more intuitive. As for optional parameters, you can return to them later 
when the minimal use case cannot fulfill your demand.


We are dealing with vectors and matrices in this course, so you will need a NumPy array to store these kind of data.
Beside the classic "np.array()" function, a few functions by specifying a shape can sometimes be very handy:
    <https://numpy.org/doc/stable/reference/routines.array-creation.html#from-shape-or-value>

After you have your arrays, you can apply operators (+, -, *, /, =, ** as exponent) to them at ease. NumPy also provides 
various methods for more advanced operations, for example, the mean value:
    <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.mean.html>
Of course, you don't have to use it. There is always more than one solution to a problem.

Quite often, you want to do something with only a part of the array, such as, a single element, a specific row/column, 
a few rows/columns, etc. This is achieved by properly indexing and slicing:
    <https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing>


After the calculation, it is the exciting moment to show them off in the plots!

The "plt.figure()" function (code already written for this exercise) will create and/or switch to a plot. Then you can call 
various Pyplot functions to draw on the plot.

To draw a line:
    <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>

To draw a histogram:
    <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html>

Optionally, you can also add title / xlabel / ylabel / legend to make your plot fancier.


Have fun!
"""


from tokenize import PlainToken
import matplotlib.pyplot as plt
import numpy as np


Nmax = 1000 # number of samples
M = 200  # number of experiments

# load exercise data
data = np.load("exercise1_dataset.npz")
u = data['u']
i = data['i']


## (a)
# YOUR CODE: calculate the values
# ===============================
# "None" is just to pass the syntax check. Replace them with your code. Don't need to stick to a single line.
# (The same holds for all the following cases)
R_SA_single = np.zeros((Nmax,1))
R_LS_single = np.zeros((Nmax,1))
R_EV_single = np.zeros((Nmax,1))
t_SA = 0.0
t_LS1 = 0.0
t_LS2 = 0.0
t_EV1 = 0.0
t_EV2 = 0.0
for k in range(0,Nmax):
    t_SA = t_SA + u[k, 0] / i[k, 0]
    t_LS1 = t_LS1 + u[k, 0] * i[k, 0]
    t_LS2 = t_LS2 + i[k, 0] * i[k, 0]
    t_EV1 = t_EV1 + u[k, 0]
    t_EV2 = t_EV2 + i[k, 0]
    R_SA_single[k, 0] = t_SA/(k+1)
    R_LS_single[k, 0] = ((1 / (k+1)) * t_LS1) / ((1 / (k+1)) * t_LS2)
    R_EV_single[k, 0] = ((1 / (k+1)) * t_EV1) / ((1 / (k+1)) * t_EV2)

x = np.arange(Nmax)
# plt.figure(figsize = (5, 2.7), layout = 'constrained')
plt.figure(1)
# YOUR CODE: draw on the plot 
plt.plot(x, R_SA_single, label = 'R_SA')
plt.plot(x, R_LS_single, label = 'R_LS')
plt.plot(x, R_EV_single, label = 'R_EV')
plt.xlabel('x label')
plt.ylabel('Resistance')
plt.title("Estimator A")
plt.legend()



## (b)
# YOUR CODE: calculate the values
R_SA = np.zeros((Nmax,M))
R_LS = np.zeros((Nmax,M))
R_EV = np.zeros((Nmax,M))

T_SA = np.zeros((1,M))
T_LS1 = np.zeros((1,M))
T_LS2 = np.zeros((1,M))
T_EV1 = np.zeros((1,M))
T_EV2 = np.zeros((1,M))
for k in range(Nmax):
    T_SA[0, 0:M] = T_SA[0, 0:M] + u[k, 0:M] / i[k, 0:M]
    T_LS1[0, 0:M] = T_LS1[0, 0:M] + u[k, 0:M] * i[k, 0:M]        
    T_LS2[0, 0:M] = T_LS2[0, 0:M] + i[k, 0:M] * i[k, 0:M]
    T_EV1[0, 0:M] = T_EV1[0, 0:M] + u[k, 0:M]
    T_EV2[0, 0:M] = T_EV2[0, 0:M] + i[k, 0:M]
    R_SA[k, 0:M] = T_SA[0, 0:M]/(k+1)
    R_LS[k, 0:M] = (T_LS1[0, 0:M]/(k+1))/(T_LS2[0, 0:M]/(k+1))
    R_EV[k, 0:M] = (T_EV1[0, 0:M]/(k+1))/(T_EV2[0, 0:M]/(k+1))

plt.figure(2)
# YOUR CODE: draw on the plot 
for j in range(M):
    plt.plot(x, R_SA[:,j], label = 'R_SA')
plt.xlabel('x label')
plt.ylabel('Resistance')
plt.title("SA")


plt.figure(3)
# YOUR CODE: draw on the plot 
for j in range(M):
    plt.plot(x, R_LS[:,j], label = 'R_LS')
plt.xlabel('x label')
plt.ylabel('Resistance')
plt.title("LS")


plt.figure(4)
# YOUR CODE: draw on the plot 
for j in range(M):
    plt.plot(x, R_EV[:,j], label = 'R_EV')
plt.xlabel('x label')
plt.ylabel('Resistance')
plt.title("EV")


## (c)
# YOUR CODE: calculate the values
R_SA_mean = np.sum(R_SA, axis = 1)
R_SA_mean = R_SA_mean/M
R_LS_mean = np.sum(R_LS, axis = 1)
R_LS_mean = R_LS_mean/M
R_EV_mean = np.sum(R_EV, axis = 1)
R_EV_mean = R_EV_mean/M

plt.figure(5)
# YOUR CODE: draw on the plot 
plt.plot(x, R_SA_mean, label = 'the mean of SA')
plt.plot(x, R_LS_mean, label = 'the mean of LS')
plt.plot(x, R_EV_mean, label = 'the mean of EV')
plt.xlabel('x label')
plt.ylabel('Resistance')
plt.title("Estimator A")
plt.legend()


## (d)
# YOUR CODE: calculate the values
R_SA_Nmax = R_SA[999,:]
R_LS_Nmax = R_LS[999,:]
R_EV_Nmax = R_EV[999,:]

plt.figure(6)
# YOUR CODE: draw on the plot 
plt.hist(R_SA_Nmax, bins = 10)
plt.xlabel("Estimated Resistance")
plt.ylabel("Counts")
plt.title("Histogram for R_SA")

plt.figure(7)
# YOUR CODE: draw on the plot 
plt.hist(R_LS_Nmax, bins = 50)
plt.xlabel("Estimated Resistance")
plt.ylabel("Counts")
plt.title("Histogram for R_LS")

plt.figure(8)
# YOUR CODE: draw on the plot 
plt.hist(R_EV_Nmax, bins = 50)
plt.xlabel("Estimated Resistance")
plt.ylabel("Counts")
plt.title("Histogram for R_EV")


# show all plots
# ===============
# If you don't see any plots but also no errors, it is very likely because a non-interactive backend is chosen by default.
# One possible solution you can try is to manually select another backend via "matplotlib.use()" function at the beginning.
# Here gives more details: <https://matplotlib.org/stable/users/explain/backends.html>
plt.show()
