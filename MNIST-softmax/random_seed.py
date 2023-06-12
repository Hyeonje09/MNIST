import numpy as np

np.random.seed(0)
print(np.random.rand(3))
np.random.seed(2)
print(np.random.rand(3))
np.random.seed(0)
print(np.random.rand(3))