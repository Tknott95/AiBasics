import numpy as np

np.random.seed(4)

"""
|| CREATING AN OBJECT ||
  i -> inputs   | when array, they become batches
  w -> weights  | will be random for this example
  b -> bias     | 

@NOTES
  transposing an array w/ numpy -> np.array().T
  dot prod w/ numpy             -> np.dot()
  random seed w/ same rand vals -> np.random.seed(int)
"""

i = [
  [1.8, 2.1, -1.2, -1.77],
  [1.75, -3.33, 2.13, -1.37],
  [-1.33, 2.22, 1.44, -1.88]
]

# CREATE LAYER CLASS HERE