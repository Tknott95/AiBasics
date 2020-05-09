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

  np.random.randn()              -> randn generates an array from the shape given of rand floats from
    a "univariate normal Gaussian distribution" | a rand float generator for arrays essentially from -1 <-> 1
    Example: 
      run    -> print(np.random.randn(2,3))
      output -> [[0.79,-0.2, 0.3], [-0.1,-0.42,0.44]]
"""

i = [
  [1.8, 2.1, -1.2, -1.77],
  [1.75, -3.33, 2.13, -1.37],
  [-1.33, 2.22, 1.44, -1.88]
]

class Layer_Dense:
  def __init__(self, _inputs, _neurons):
    self.weights = np.random.randn(_inputs, _neurons)