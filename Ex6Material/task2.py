"""
This is the template for coding problems in exercise sheet 6, task 2.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.

Have fun!
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize


#load data
data = np.load("exercise6_task2_data.npz")
v_w = data['v_w']

## (a) and (b): On paper

## (c) Paper: Take a look at the histogram of the probability density function of the provided wind speed data of the German Weather Service taken on Feldberg (the code for plotting the histogram is already provided in the solution template). What do you observe?

# Plot histogram
bins = np.ceil(v_w.max()).astype("int")  # number of bars for histogram
plt.figure(1)
plt.hist(v_w, bins, density=True, edgecolor="black", linewidth=0.5)
plt.title('Histogram of the wind speeds at Feldberg')
plt.ylabel('frequency of occurence')
plt.xlabel('v_w [m/s]')


## (d) Solve the minimization problem obtained in Task (b) to estimate λ and k
N = len(v_w) # Number of wind samples
e = 1e-10

# define the objective function
def objectiveFunction(theta):
   """
   The objective function of the ML problem.
   Returns the value of the objective function for a given parameter theta.
   """
   # theta = [k, lambda]

   ######## YOUR CODE ########
   # value = lambda x: 
   # return (theta[0]**N/theta[1]*N)*np.prod((v_w/theta[1])**(theta[0]-1))*np.exp(-np.sum((v_w/theta[1])**theta[0]))
   # value = np.sum((v_w/theta[1])**theta[0]) - (theta[0]-1)*np.sum(np.log(v_w/theta[1])) - N*np.log(theta[0]) + N*np.log(theta[1])
   # value = (v_w/theta[1])**theta[0] - (theta[0]-1)*np.log(v_w/theta[1]) - N*np.log(theta[0]) + N*np.log(theta[1])
   # logL = N*np.log(theta[0]) - N*np.log(theta[1]) - np.sum((v_w/theta[1])**theta[0]) + (theta[0]-1)*np.sum(np.log(v_w/theta[1]))
   logL = -N*np.log(theta[0]) + N*theta[0]*np.log(theta[1]) + np.sum((v_w/theta[1])**theta[0]) - (theta[0])*np.sum(np.log(v_w))
   # return value
   return logL
   ###########################

# cons = ({'type': 'ineq', 'fun': lambda theta: theta[0] - 0 },
#         {'type': 'ineq', 'fun': lambda theta: theta[1] - 0 })

bnd = ((0, None), (0, None))


# give lambda and k initial values for scipy's minimize function
lambda_0 = 2
k_0 = 10
theta0 = [k_0, lambda_0]

# solve the minimization problem with scipy's minimize function
######## YOUR CODE ########
minResult = minimize(fun=lambda theta, data: objectiveFunction(theta), x0 = theta0, args=(v_w), bounds=bnd)
###########################

# extract the solution
thetaStar = minResult.x # the field .x holds the minimizer
kStar = thetaStar[0]
lambdaStar = thetaStar[1]
# print("kStar, lambdaStar: ", kStar, lambdaStar)

# define the weibull distribution 
def weibullPDF(v,k,lam):
   """
   The probability density function for a weibull distribution.
   Returns the probability of a wind speed v for given distribution parameters k and lam.
   """
   ######## YOUR CODE ########
   # for i in range(N):
   #    if v[i] > 0:
   #       return (k/lam)*((v[i]/lam)**(k-1))*np.exp(-(v[i]/lam)**k)
   #    else:
   #       return 0
   # # return None
   # if v > 0:
   return (k/lam)*((v/lam)**(k-1))*np.exp(-(v/lam)**k)
   # else:
   #    return 0
   ###########################

# for plotting purposes only
v_plotting = np.linspace(0, 30, 1000) # array of wind speeds to plot continuos distribution

## Plot fitted pdf into histogram
plt.figure(2)
plt.hist(v_w, bins, density=True, edgecolor="black", linewidth=0.5)
plt.plot(v_plotting, weibullPDF(v_plotting, thetaStar[0], thetaStar[1]), 'r', lw=3)
plt.legend(['Histogram of v_w', 'fitted Weibull distribution'])
plt.title('PDF of the wind speed')
plt.ylabel('p(v_w)')
plt.xlabel('v_w [m/s]')


## (d) Using the data of the power curve of a specific wind turbine given in table (1), compute the expected value of the turbine power on the studied location. Use the trapezoidal rule for integration to compute the expected value of the power

# power-curve values from the table
windSpeed = np.linspace(0,26, 27)
power = [0, 0, 3, 25, 82, 174, 321, 532, 815, 1180, 1580, 1900, 2200, 2480, 2700, 2850, 2950, 3020, 3020, 3020, 3020, 3020, 3020, 3020, 3020, 3020, 0]

######## YOUR CODE ########
# compute the probabilities for the windSpeeds of the table
windSpeedProbability = (kStar/lambdaStar)*((windSpeed/lambdaStar)**(kStar-1))*np.exp(-(windSpeed/lambdaStar)**kStar)
###########################

# numerically compute the integral in the expected value formula using the trapezoid rule
expectedPower = 0    
for i in range(26):
   ######## YOUR CODE ########
   expectedPower = expectedPower + 0.5*(power[i]*windSpeedProbability[i]+power[i+1]*windSpeedProbability[i+1])
   ###########################

# print the solution
print("k* = %0.3f | lambda* = %0.3f | E(Power)* = %0.3f \n" % (kStar, lambdaStar, expectedPower))

plt.show()
