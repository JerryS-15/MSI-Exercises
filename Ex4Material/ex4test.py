import numpy as np

# data = np.load('exercise4_dataset.npz')
# # print(data.files, "\n")
# U = data['U']
# I = data['I']
# N_e = np.size(I, 0)  # number of students/experiments
# N_m = np.size(I, 1)  # number of measurements per experiment

# # print("Data U: ", u[999,:])
# print("Number of Students/Experiments: ", N_e)
# print("Number of mearsurements per experiment: ", N_m)
# print("\n")
# print("Data I: ", I[0,:])
# print("\n")
# print("Data U: ", U[0,:])

m = np.array([1.21, -6.66])

print(np.dot(m,m.reshape((2,1))))