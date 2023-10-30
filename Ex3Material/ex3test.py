import matplotlib.pyplot as plt
import numpy as np


## load data
data = np.load("exercise3_task4_dataset.npz")
i1 = data['i1']
u1 = data['u1']
i2 = data['i2']
u2 = data['u2']
N = 2000

print("Data i1: ", i1)
print("\n")
print("Data u1: ", u1)
print("\n")
# c1 = np.ones((N))
# s = np.column_stack((c1, i1))

# x = np.dot(s.T, s)
# X = np.matrix(x)
# x = np.linalg.solve(s, u1)
# print(x)
# print("\n")
# print(X.I)
# print("\n")
c1 = np.ones((N))
Phi1 = np.column_stack((c1, i1))
Phi1_plus = np.dot(Phi1.T, Phi1)
theta_star1 = np.dot(np.linalg.pinv(Phi1_plus), Phi1.T)
theta_star1 = np.dot(theta_star1, u1)
print(theta_star1)