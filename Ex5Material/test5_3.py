import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = np.load("exercise5_task3_data.npz")
X = data['X'] # X is of shape (N_e, N_m)
Y = data['Y'] # Y is of shape (N_e, N_m)

alphas = [0, 1e-6, 1e-5, 1]
N_alpha = len(alphas) 

# print("X: ", X, "\nY: ", Y)
# parameter = np.polyfit(X[0,:], Y[0,:], 7)

phi = np.array([[X[0,0]**7, X[0,0]**6, X[0,0]**5, X[0,0]**4, X[0,0]**3, X[0,0]**2, X[0,0]**1, X[0,0]**0]])

# Phi = np.ones((9, 8))
# # Phi = np.dot(Phi, parameter.T)
# for i in range(8):
#     for j in range(9):
#         Phi[j][i] = parameter[i]*(X[0,j]**(i+1))
Phi = np.ones((9, 8))
for i in range(8):
    for j in range(9):
        Phi[j][i] = (X[0,j]**(7-i))


I_Matrix = np.identity(8)

print("phi: ", phi)
print("\nPhi: ", Phi)

# for i in range(N_alpha):
#     alpha = alphas[i]

#     ######## YOUR CODE ########
#     # solve the LLS problem to find the parameters of the fit
#     theta = np.dot(np.dot(np.linalg.pinv(np.dot(Phi.T, Phi) + alpha*I_Matrix), Phi.T), Y[0,:])
#     print(i, " ", theta)