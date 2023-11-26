"""
This is the template for coding problems in exercise sheet 5, task 3.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.

Have fun!
"""

import matplotlib.pyplot as plt
import numpy as np


# Load data
data = np.load("exercise5_task3_data.npz")
X = data['X'] # X is of shape (N_e, N_m)
Y = data['Y'] # Y is of shape (N_e, N_m)

# array of the alphas
alphas = [0, 1e-6, 1e-5, 1]

# order of the fitting polynomials
order = 7

# derive useful constants
N_alpha = len(alphas) #
N_theta = order + 1  # number of parameters
N_e = np.size(X, 0)  # number of students/experiments
N_m = np.size(X, 1)  # number of measurements per experiment

## x and Phi used for plotting the polynomials
x_plotting = np.linspace(0, 1, 200) 
Phi_plotting = np.ones((200, order+1))
for o in range(order+1):
    Phi_plotting[:, o] = x_plotting.T**o


## Single experiment
plt.figure(1)
plt.plot(X[0,:], Y[0,:], "x")
plt.xlim([0, 1])
plt.ylim([-7, 7])
plt.xlabel("X")
plt.ylabel("Y")

# (a) For α ∈ {0, 10−6, 10−5, 1}, fit a polynomial of order 7 to the data of the first experiment. Plot the data and the fitted polynomials.
######## YOUR CODE ########
# define the regressor matrix of the LLS fit

# parameter = np.polyfit(X[0,:], Y[0,:], 7)
# Phi = np.array([[X[0,0]**7, X[0,0]**6, X[0,0]**5, X[0,0]**4, X[0,0]**3, X[0,0]**2, X[0,0]**1, X[0,0]**0],
# [X[0,1]**7, X[0,1]**6, X[0,1]**5, X[0,1]**4, X[0,1]**3, X[0,1]**2, X[0,1]**1, X[0,1]**0],
# [X[0,2]**7, X[0,2]**6, X[0,2]**5, X[0,2]**4, X[0,2]**3, X[0,2]**2, X[0,2]**1, X[0,2]**0],
# [X[0,3]**7, X[0,3]**6, X[0,3]**5, X[0,3]**4, X[0,3]**3, X[0,3]**2, X[0,3]**1, X[0,3]**0],
# [X[0,4]**7, X[0,4]**6, X[0,4]**5, X[0,4]**4, X[0,4]**3, X[0,4]**2, X[0,4]**1, X[0,4]**0],
# [X[0,5]**7, X[0,5]**6, X[0,5]**5, X[0,5]**4, X[0,5]**3, X[0,5]**2, X[0,5]**1, X[0,5]**0],
# [X[0,6]**7, X[0,6]**6, X[0,6]**5, X[0,6]**4, X[0,6]**3, X[0,6]**2, X[0,6]**1, X[0,6]**0],
# [X[0,7]**7, X[0,7]**6, X[0,7]**5, X[0,7]**4, X[0,7]**3, X[0,7]**2, X[0,7]**1, X[0,7]**0],
# [X[0,8]**7, X[0,8]**6, X[0,8]**5, X[0,8]**4, X[0,8]**3, X[0,8]**2, X[0,8]**1, X[0,8]**0]])

Phi = np.ones((9, 8))
for i in range(8):
    for j in range(9):
        Phi[j][i] = X[0,j]**i
###########################

# empty arrays to store the results
thetas_single = np.zeros((N_alpha, N_theta))
thetas_single_norm = np.zeros((N_alpha, 1))
R_squared = np.zeros((N_alpha, 1))

I_Matrix = np.identity(8)

temp_y = 0
for j in range(9):        
    temp_y = temp_y + Y[0, j] * Y[0, j]

# Y1 = np.zeros((9, 1))
# for j in range(9):
#     Y1[j, 0] = Y[0, j]

# iterate the alphas
for i in range(N_alpha):
    alpha = alphas[i]

    ######## YOUR CODE ########
    # solve the LLS problem to find the parameters of the fit
    theta = np.linalg.inv(np.dot(Phi.T, Phi) + alpha*I_Matrix)
    theta = np.dot(np.dot(theta, Phi.T), Y[0,:])
    # theta = np.dot(np.dot(np.linalg.inv(np.dot(Phi.T, Phi) + alpha*I_Matrix), Phi.T), Y[0,:])
    ###########################
    # print(theta)

    # save values
    thetas_single[i, :] = theta

    ######## YOUR CODE ########
    # (b) For experiment 1 and for each α, compute the L2-norm of the estimated parameters
    # compute the norm and the R squared value
    # thetas_single_norm[i] = np.sqrt(np.dot(theta.T, theta))
    thetas_single_norm[i] = np.linalg.norm(theta, ord=2)
    ###########################

    ######## YOUR CODE ########
    # (c) To compare the goodness of fit, compute the R2 values for each of the three fits obtained for experiment 1.
    R_squared[i] = 1 - np.dot((Y[0,:] - np.dot(Phi, theta)).T, (Y[0,:] - np.dot(Phi, theta)))/temp_y
    ###########################

    # print(np.dot(Phi, theta))

    # add the plot of the fit to the figure
    plt.plot(x_plotting, Phi_plotting@theta)

# add a legend
plt.legend(["data", "no regularization",
            "small regularization", "strong regularization",
            "very strong regularization"],
            loc="lower left")

### Test ###
print("thetas_single: ", thetas_single)
print("thetas_single_norm: ", thetas_single_norm)
print("R_squared: ", R_squared)


### (d) ####
# empty arrays to store the results
thetas = np.zeros((N_alpha, N_e, N_theta))
thetas_mean = np.zeros((N_alpha, N_theta))

# titles of the subplots
titles = ("no regularization", "small regularization", "strong regularization", "very strong")

plt.figure(2,figsize=(12,6))

# iterate the alphas
for i in range(N_alpha):
    alpha = alphas[i]
    
    # (d) For each α and each experiment, fit a polynomial of order 7. For each α, plot the fitted polynomials in a subplot

    # create a subplot 
    plt.subplot(N_alpha, 2, 2*i+1)
    plt.title(titles[i])
    plt.ylim([-6, 6])
    plt.xlim([0, 1])
    
    # iterate the experiments
    for k in range(N_e):

        ######## YOUR CODE ########
        # define the regressor matrix of the LLS fit
        Phi = np.ones((9, 8))
        for m in range(8):
            for j in range(9):
                Phi[j][m] = X[k,j]**m
        # Phi = 

        # solve the LLS problem to find the parameters of the fit
        theta = np.dot(Phi.T, Phi) + alpha * I_Matrix
        theta = np.dot(np.linalg.inv(theta), Phi.T)
        theta = np.dot(theta, Y[k,:])
        ###########################
        
        # store the result
        thetas[i,k,:] = theta

        # plot the fit
        plt.plot(x_plotting, Phi_plotting@theta, "-");  
    

    # Compute the average parameter vector for each α and plot the polynomial obtained from the averaged parameter vector.
    plt.subplot(N_alpha, 2, 2*i+2)
    plt.title(titles[i])
    plt.ylim([-6, 6])
    plt.xlim([0, 1])

    ######## YOUR CODE ########
    # compute the average parameter vector
    theta_mean_unit = np.zeros((N_e, 8))
    for k in range(N_e):
        theta_mean_unit[k] = thetas[i,k,:]
    theta_mean = np.mean(theta_mean_unit, axis=0)
    ###########################
    
    # store the results
    thetas_mean[i,:] = theta_mean
    
    # plot the average fit
    plt.plot(x_plotting, Phi_plotting@theta_mean, "-")

plt.tight_layout()
plt.show()
