import numpy as np
import matplotlib.pyplot as plt

# load the data
data = np.load("exercise7_task1_refSol.npz")
print(data.files)
theta_e = data['predictions']
print(theta_e)
# u = data['u']
# XYm = data['XYm']

# l_u = u.shape[0]
# l_XYm = XYm.shape[0]

# dT = 0.0159 # sampling time
# N = XYm.shape[0] # number of datapoints
# t = dT * np.arange(N) # array of all sampling times

# order = 4
# N = XYm.shape[0]

# theta_a = np.zeros((order+1, 2)) 
# ######## YOUR CODE ########
# # compute the LLS estimate
# # Phi = np.ones((N, 5, 2))
# # Phi = np.ones((N, 5))
# # for i in range(5):
# #     for j in range(N):
# #         Phi[j][i] = XYm[j]**i
# #         Phi[j][i] = XYm[j]**i

# # temp_theta = np.dot(np.linalg.inv(np.dot(Phi.T, Phi)), Phi.T)
# Phi = np.power.outer(t, range(order+1))
# nextPhi = Phi[1,:]
# print(nextPhi)

# print(np.dot(nextPhi, nextPhi.T))

# # print("\nu: \n", u)
# # print("\nXYm: \n", XYm)
# # print("Length u, XYm: ", l_u, XYm[0, 1])