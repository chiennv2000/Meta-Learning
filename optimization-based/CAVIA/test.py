import numpy as np

x = np.zeros((3, 4))

a = np.array([1, 2, 3])
a = np.stack([a] * 3)

x = np.concatenate((x, a), axis=1)
y = x[:, -3:]
print(np.mean(y, axis=1))