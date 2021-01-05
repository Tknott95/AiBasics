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

  class Layer_Dense:
    def __init__(self, _numOfInputs, _numOfNeurons):
      self.weights = 0.10 * np.random.randn(_numOfInputs, _numOfNeurons)
      self.biases = np.zeros((1, _numOfNeurons))
    def next(self, _inputs): # Usually called "forward"
      self.output = np.dot(_inputs, self.weights) + self.biases

  class Activation_Softmax:
    def next(self, _inputs):
      exponential_values = np.exp(_inputs - np.max(_inputs, axis=1, keepdims=True))
      normalized_values = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
      self.output = normalized_values

  layer1 = Layer_Dense(4, 8)
  mySoftmax = Activation_Softmax()
  # !!! Softmax brought in by hand yet not passed into layer 2 after applied like you normally would for visuals
 
  layer2 = Layer_Dense(8, 6)
  layer3 = Layer_Dense(6, 4)

  layer1.next(i)
  mySoftmax.next(layer1.output)

  print('\n\n layer1: \n',layer1.output)
  layer2.next(layer1.output)
  print('\n\n layer2: \n',layer2.output)
  layer3.next(layer2.output)
  print('\n\n layer3: \n', layer3.output)

  print('\n\n layer1 after Softmax: \n', mySoftmax.output)

main()
