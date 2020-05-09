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
    using 0.10 * np.random.randn() for better "mock data"

    Example: 
      run    -> print(np.random.randn(2,3))
      output -> [[0.79,-0.2, 0.3], [-0.1,-0.42,0.44]]

      run    -> print(np.random.randn(4,3))
      output -> [
        [ 0.05056171  0.49995133 -0.99590893]
        [ 0.69359851 -0.41830152 -1.58457724]
        [-0.64770677  0.59857517  0.33225003]
        [-1.14747663  0.61866969 -0.08798693]]

  np.zeros()                     -> returns a tuple of an array of the shape
    Example:
      run    -> print(np.zeros(4))
      output -> [0. 0. 0. 0.]

      run    -> print(np.zeros((1, 4)))
      output -> [[0. 0. 0. 0.]]

      run    -> print(np.zeros((2, 4)))
      output -> [
        [0. 0. 0. 0.]
        [0. 0. 0. 0.]]
"""
def main():
  i = [
    [1.8, 2.1, -1.2, -1.77],
    [1.75, -3.33, 2.13, -1.37],
    [-1.33, 2.22, 1.44, -1.88]
  ]

  print(np.zeros((2, 4)))

  class Layer_Dense:
    def __init__(self, _inputs, _neurons):
      self.weights = 0.10 * np.random.randn(_inputs, _neurons)
      self.biases = np.zeros((1, _neurons))
    def next(self):
      pass

main()