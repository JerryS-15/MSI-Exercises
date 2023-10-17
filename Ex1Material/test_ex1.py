import numpy as np

data = np.load('//Users//jerrysong//vscode//msi//Ex1Material//exercise1_dataset.npz')
# print(data.files, "\n")
u = data['u']
i = data['i']

print("Data u: ", u[999,:])
print("\n")
print("Data i: ", i[999,:])
print("\n")
# s = np.zeros((1, 200))
# for j in range(0, 10):
#     s[0, 0:10] = s[0, 0:10] + u[j, 0:10]/i[j, 0:10]
#     print(j, "s: ", s[0, 0:10]/(j+1))

# for j in range(0, 10):
#     print(j, ": ", u[j, 0:10]/i[j, 0:10])