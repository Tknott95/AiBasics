import numpy as np

i = [1, 2, 3, 4]
w = [[-0.4, 0.7, 0.2, 0.4], [1.4, -0.8, -0.4, 0.4]]
b = [0.2, 0.4]

output = np.dot(w, i) + b

print(output)

