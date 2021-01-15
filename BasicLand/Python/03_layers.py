import numpy as np

np.random.seed(4)

"""
|| Creating an obj w/ random generators for neural net theory ||
  i -> inputs   | when array, they become batches
  w -> weights  | will be random for this example
  b -> bias     | 

@NOTES
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

  @Custom
    Layer_Dense(<numberOfInputs/features, so 4>, <num of neurons, can be any #>)
     next layer input = lastLayer.output
"""

def main():
  i = [
    [1.8, 2.1, -1.2, -1.77],
    [1.75, -3.33, 2.13, -1.37],
    [-1.33, 2.22, 1.44, -1.88]
  ]

  class LayerDense:
    def __init__(self, _numOfInputs, _numOfNeurons):
      self.weights = 0.10 * np.random.randn(_numOfInputs, _numOfNeurons)
      self.biases = np.zeros((1, _numOfNeurons))
    def next(self, _inputs):
      self.output = np.dot(_inputs, self.weights) + self.biases


  layer1 = LayerDense(4, 8)
  layer2 = LayerDense(8, 6)
  layer3 = LayerDense(6, 4)

  layer1.next(i)
  print('\n\n layer1: \n',layer1.output)
  layer2.next(layer1.output)
  print('\n\n layer2: \n',layer2.output)
  layer3.next(layer2.output)
  print('\n\n layer3: \n', layer3.output)

main()