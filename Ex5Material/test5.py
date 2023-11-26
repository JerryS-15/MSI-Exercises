import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

y = np.arange(-3,6) # measurement vector
phi = np.array([[1/6,1/6]]) # regression vector
Phi = np.repeat(phi,9,axis=0) # regressor matrix

x1 = np.dot(Phi.T, Phi)
x = np.dot(np.linalg.pinv(np.dot(Phi.T, Phi)), Phi.T)
u, s, vt = np.linalg.svd(Phi)

# S = np.zeros([9, 2])
# for i in range(2):
#     S[i][i] = s[i]

S0 = np.zeros([2, 9])
for i in range(2):
    S0[i][i] = s[i]/(s[i] * s[i] + 0.2)
I_Matrix = np.identity(2)

# x3 = np.dot(np.dot(u, S), vt)
# x2 = np.dot(np.dot(np.linalg.inv(np.dot(Phi.T, Phi) + 0.2*I_Matrix), Phi.T), y)
# x3 = np.dot(np.dot(np.dot(vt.T, np.linalg.inv(np.dot(S.T, S) + 0.2*I_Matrix)), S.T), u.T)



# print("u: ", u,"\ns: ", S,"\nv: ", vt.T)
# print("S0: ", S0)
print("y: ", y)