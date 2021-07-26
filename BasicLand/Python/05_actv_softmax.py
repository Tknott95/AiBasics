import numpy as np

"""

  ACTIVATION FUNCTION 01 - Softmax

  - only applying to the first layer for an ouptu comparison for visuals.
  - will not be passing the output after an activation function pass into layer 2. 
  - complexity right meow is not needed rather visuals for learning each "module"

"""
 
np.random.seed(4)

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
    def forward(self, _inputs):
      self.output = np.dot(_inputs, self.weights) + self.biases

  class ActivationSoftmax:
    def forward(self, _inputs):
      exponential_values = np.exp(_inputs - np.max(_inputs, axis=1, keepdims=True))
      normalized_values = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
      self.output = normalized_values

  layer1 = LayerDense(4, 8)
  mySoftmax = ActivationSoftmax()
 
  layer2 = LayerDense(8, 6)
  layer3 = LayerDense(6, 4)

  layer1.forward(i)
  mySoftmax.forward(layer1.output)

  print('\n\n layer1: \n',layer1.output)
  layer2.forward(layer1.output)
  print('\n\n layer2: \n',layer2.output)
  layer3.forward(layer2.output)
  print('\n\n layer3: \n', layer3.output)

  print('\n\n layer1 after Softmax: \n', mySoftmax.output)

main()
