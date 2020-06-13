import pandas as pd
import numpy as np

v1 = pd.np.array([1, 2, 3])
v2 = pd.np.array([2, 3, 4])
print("The dot product of the two vectors is:")
print(v1.dot(v2))

print("Manual computation of the dot product:")
print(sum([x1 * x2 for x1, x2 in zip(v1, v2)]))
