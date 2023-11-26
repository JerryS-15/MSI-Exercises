"""
This is the template for coding problems in exercise sheet 7, task 1.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.

HINT: In this exercise we have two different parameter vectors for the linear fits of both the X- and Y-Coordinate. We collect in a PARAMETER MATRIX theta=[theta_x, theta_y]. It is possible to do the math operations with this matrix instead of (as before) just the single vectors.

Have fun!
"""

import numpy as np
import matplotlib.pyplot as plt

# load the data
data = np.load("exercise7_task1_data.npz")
XYm = data['XYm']

dT = 0.0159 # sampling time
N = XYm.shape[0] # number of datapoints
t = dT * np.arange(N) # array of all sampling times

## (a) Fit a 4-th order polynomial through the data using linear least-squares. Plot the data and the fit for the X- and Y-coordinate. 

# order of the polynomial
order = 4

# empty array to store the parameter matrix theta for bot the X and Y coordinate
theta_a = np.zeros((order+1, 2)) 
######## YOUR CODE ########
# compute the LLS estimate
Phi = np.ones((N, 5))
# Phi1 = np.ones((N, 5))
for i in range(5):
    for j in range(N):
        Phi[j][i] = t[j]**i
        # Phi[j][i] = XYm[j]**i
temp_theta = np.dot(np.linalg.inv(np.dot(Phi.T, Phi)), Phi.T)
theta_a[:,0] = np.dot(temp_theta, XYm[:,0])
theta_a[:,1] = np.dot(temp_theta, XYm[:,1])
###########################

# compute the fitting polynomials
polyx = Phi@theta_a[:,0]
polyy = Phi@theta_a[:,1]

# Figure 1: Plot solution for both coordinates seperately
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t, XYm[:,0], "o", markerfacecolor="none")
plt.plot(t, polyx)

plt.subplot(2,1,2)
plt.plot(t, XYm[:,1], "o", markerfacecolor="none")
plt.plot(t, polyy)

# Figure 2: Plot the data and LLS solution in both coordinates
plt.figure(2)
plt.plot(XYm[:,0], XYm[:,1], "ok", markerfacecolor="none")
plt.plot(XYm[:,0], XYm[:,1], "-k")
plt.plot(polyx, polyy, "-b")

## (b) Implement the RLS algorithm as described in the script (Check section 5.3.1) to estimate 4-th order polynomials to fit the data. Do not use forgetting factors yet. Plot the result against the data on the same plot as the previous question.

# parameter matrix for the RLS estimates, initial value is zero
theta_b = np.zeros((order+1,2))
# small initial value for the matrix Q
Q = np.eye(order+1) * 1e-10

# regressor matrix
Phi = np.power.outer(t, range(order+1))

######## YOUR CODE ########
# compute the recursive RLS estimate
for k in range(N):
    nextPhi = np.zeros((5, 2))
    for j in range(5):
        nextPhi[j][0] = t[k]**j
        nextPhi[j][1] = t[k]**j
    # nextPhi = [[t[k]**0, t[k]**1, t[k]**2, t[k]**3, t[k]**4]]
    nextQ = Q + np.dot(nextPhi, nextPhi.T)
    nextTheta = theta_b + np.linalg.inv(nextQ)@nextPhi@(XYm[k]-nextPhi.T@theta_b)
    
    theta_b = nextTheta
    Q = nextQ
##########################

# plot the RLS solution into # Figure 2
rlsFit = Phi@theta_b
plt.figure(2)
plt.plot(rlsFit[:,0], rlsFit[:,1], "-g")
plt.legend([
    "Robot location samples", "Interpolated Robot location",
    "4th order polynomial fit", "4th order polynomial fit, recursive implementation"
])


## (c) Add a forgetting factor alpha to your algorithm and try different values for \alpha.  Plot the results against the data.

# Figure 3: RLS with different alphas
plt.figure(3)
plt.plot(XYm[:,0], XYm[:,1], "ok", markerfacecolor="none", label="robot location samples")

alphas = [0.7, 0.9]

# empty array to store the parameter matrices
# f.e theta_c[:,0,0] containt the parameter vector of size (order+1) for the x-coordinates, for the first alpha
theta_c = np.zeros((order+1,2,len(alphas)))

# regressor matrix
Phi = np.power.outer(t, range(order+1))

for n in range(len(alphas)):
    # small initial value for the matrix Q
    Q = np.eye(order+1) * 1e-10

    ######## YOUR CODE ########
    # compute the recursive RLS estimate
    for k in range(N):
        nextPhi = np.zeros((5, 2))
        # for j in range(5):
        #     nextPhi[j][0] = t[k]**j
        #     nextPhi[j][1] = t[k]**j
        nextPhi[:,0] = Phi[k,:]
        nextPhi[:,1] = Phi[k,:]
        nextQ = alphas[n]*Q + np.dot(nextPhi, nextPhi.T)
        nextTheta = theta_c[:,:,n] + np.linalg.inv(nextQ)@nextPhi@(XYm[k]-nextPhi.T@theta_c[:,:,n])
        
        theta_c[:,:,n] = nextTheta
        Q = nextQ
    ##########################

    # plot the fitted curve
    rlsFit = Phi@theta_c[:,:,n]
    plt.plot(rlsFit[:,0], rlsFit[:,1], label=r"fit with $\alpha = %.3f$" % alphas[n])

plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.legend()


## (d) Compute the one-step-ahead prediction at each point (i.e. extrapolate your polynomial fit to the next time step). We also provided code to plot the 1-sigma confidence ellipsoid around this point, and the data.
plt.figure(4)

# regressor matrix
Phi = np.power.outer(t, range(order+1))

# two initial values for the matrix Q for each coordinate
Qx = np.eye(order+1) * 1e-10
Qy = Qx.copy()

# two different forgetting factors each coordinate
alphax = 0.85
alphay = 0.78

# array to store the one-step-ahead predictions
predictions = np.zeros((N, 2))

# iterates where the one-step-ahead prediction is plotted
ks = [3, 20, 50, 75]
k_iter = iter(range(4)) # to keep track of the current subplot

# empty array to store the parameter matrix
theta_e = np.zeros((order+1, 2))

# empty array to store the next parameter matrix
nextTheta = np.zeros_like(theta_e)

# compute the one-step-ahead prediction for each step

nextPhi = np.zeros((5, 1))
for k in range(N):

    ########### YOUR CODE #############
    # compute the recursive RLS update for each coordinate seperately
    nextPhi[:,0] = Phi[k,:]
    nextQx = alphax*Qx + np.dot(nextPhi, nextPhi.T)
    nextQy = alphay*Qy + np.dot(nextPhi, nextPhi.T)
    nextTheta[:,0] = theta_e[:,0] + np.linalg.inv(nextQx)@nextPhi@(XYm[k,0] - nextPhi.T@theta_e[:,0])
    nextTheta[:,1] = theta_e[:,1] + np.linalg.inv(nextQy)@nextPhi@(XYm[k,1] - nextPhi.T@theta_e[:,1])
    # temp = np.zeros((1,1))
    # temp[0][0] = XYm[k,0] - nextPhi.T@theta_e[:,0]
    # print("temp: ", temp)
    # temp2 = np.linalg.inv(nextQx)@nextPhi
    # print("temp2: ", temp2)
    # nextTheta[:,0] = theta_e[:,0] + temp2@temp
    # print(nextPhi)

    
    # compute the one-step-ahead prediction
    # nextPoint = [nextPhi[:,0]@nextTheta[:,0],nextPhi[:,0]@nextTheta[:,1]]
    nextPoint = [nextPhi[:,0]@theta_e[:,0],nextPhi[:,0]@theta_e[:,1]]
    
    #################################
    
    # save the prediction
    predictions[k,:] = nextPoint
    # predictions[0,:] = [0,0]

    # compute the fit for this iterate
    rlsFit = Phi[0:k, :] @ theta_e
    
    # transfer values to next iteration
    theta_e = nextTheta
    Qx = nextQx
    Qy = nextQy
    
    # plot some of the predictions
    if k in ks:
        # create a nice plot of the prediction + confidence elipsoid
        plt.subplot(len(ks)//2, 2, next(k_iter)+1) 
        plt.plot(XYm[:,0], XYm[:,1], "ok")  
        plt.plot(rlsFit[:,0], rlsFit[:,1], "-")  
        plt.plot(XYm[0:k, 0], XYm[0:k, 1], "xk")
        plt.plot(XYm[k, 0], XYm[k, 1], "xr")
        plt.plot(nextPoint[0], nextPoint[1], "x")
        
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        
        sigx = nextPhi.T @ np.linalg.solve(nextQx, nextPhi)
        sigy = nextPhi.T @ np.linalg.solve(nextQy, nextPhi)

        sigx = sigx[0, 0]
        sigy = sigy[0, 0]

        sig = np.array([
            [sigx, 0],
            [0, sigy],
        ])

        # Confidence ellipsoids
        [D,V] = np.linalg.eig(sig)  # eigenvalues and eigenvectors

        xy = np.vstack((np.cos(np.linspace(0, 2*np.pi, 50)), np.sin(np.linspace(0, 2*np.pi, 50))))
        xy_ellipse = np.outer(nextPoint, np.ones(50)) + V@np.sqrt(np.diag(D))@xy
        
        plt.plot(xy_ellipse[0,:], xy_ellipse[1,:])
        
        plt.legend(
            ["all data", "current fit", "available data",
            "next datapoint", "prediction", "confidence ellipsoid"],
            loc="upper right"
        )
        plt.title("timestep %d" % k)

plt.show()
