import numpy as np

"""
i -> inputs   | when array, they become batches
w -> weights  | need to transpose to have dotProd work proper, i_size(3,4) w_size(4,3) -> after transpose
b -> bias     | 

@NOTES
  transposing an array w/ numpy -> np.array().T
  dot prod w/ numpy             -> np.dot()
"""

i = [
  [1.8, 2.1, -1.2, -1.77],
  [1.75, -3.33, 2.13, -1.37],
  [-1.33, 2.22, 1.44, -1.88]
]

w = [
  [-0.4, 0.7, 0.2, -0.43],
  [0.45, -0.82, -0.44, 0.38],
  [-0.88, -0.75, 0.91, 0.11]
]

b = [0.2, 0.4, 2]

output = np.dot(w, np.array(i).T) + b

print(output)
