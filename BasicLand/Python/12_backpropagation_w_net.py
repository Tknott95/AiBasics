import numpy as np
# Pulling in nnfs data
import nnfs
from nnfs.datasets import spiral_data
# import matplotlib.pyplot as plt

class Layer_Dense:
    def __init__(self, _numOfInputs, _numOfNeurons):
      self.weights = 0.10 * np.random.randn(_numOfInputs, _numOfNeurons)
      self.biases = np.zeros((1, _numOfNeurons))
    def forward(self, _inputs):
      self._inputs = _inputs
      self.output = np.dot(_inputs, self.weights) + self.biases
    def backward(self, dValues): # dValues is derivativeValues
      # My gradients on params
      self.dWeights = np.dot(self._inputs.T, dValues)
      self.dBiases = np.sum(dValues, axis=0, keepdims=True)
      # My gradient on values
      self.dInputs = np.dot(dValues, self.weights.T)


class Activation_ReLU:
  def forward(self, _inputs):
    self.output = np.maximum(0, _inputs)
  def backward(self, dValues):
    self.dInputs = dValues.copy() 
    self.dInputs[self._inputs <= 0] = 0     # Zero gradient where input values were negative


class Activation_Softmax:
  def forward(self, _inputs):
    exponential_values = np.exp(_inputs - np.max(_inputs, axis=1, keepdims=True))
    normalized_values = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
    self.output = normalized_values
  def backward(self, dValues):
    self.dInputs = np.empty_like(dValues) # unitialized array
    
    for index, (singleOutput, singleDValues) in \
                enumerate(zip(self.output, dValues)):
                singleOutput = singleOutput.reshape(-1, 1)
                jacobianMatrix = np.diagflat(singleOutput) - \
                np.dot(singleOutput, singleOutput.T)
                self.dInputs[index] = np.dot(jacobianMatrix, singleDValues) 

class Loss:
  def calculate(self, output, y):
    sampleLosses = self.forward(output, y)
    dataLoss = np.mean(sampleLosses)

    return dataLoss

class CategoricalCrossEntropyLoss(Loss):
  def forward(self, yPrediction, yTrue):
    numOfSamples = len(yPrediction)
    """ NNFS note on line/code below
      Clip data to prevent division by 0. 
      Clip both sides to not drag mean towards any value """
    yPredictionClipped = np.clip(yPrediction, 1e-7, 1 - 1e-7)

    if len(yTrue.shape) == 1:
      correctConfidences = yPredictionClipped[
        range(numOfSamples),
        True
      ]
    elif len(yTrue.shape) == 2:
      correctConfidences = np.sum(
        yPredictionClipped*yTrue,
        axis=1
      )
    
    negativeLogLikelihoods = -np.log(correctConfidences)
    return negativeLogLikelihoods
  def backward(self, dValues, yTrue):
    samples = len(dValues)
    labels = len(dValues[0])

    if len(yTrue.shape) == 1: # one hot vector conversion if labels are "sparse", nnfs wording
      yTrue = np.eye(labels)[yTrue]
    
    self.dInputs = -yTrue / dValues # calculating gradient
    self.dInputs = self.dInputs / samples # normalizing gradient

class ActivationSoftmaxLossCategoricalCrossEntropy():
  def __init__(self):
    self.activation = Activation_Softmax()
    self.loss = CategoricalCrossEntropyLoss()
  def forward(self, _inputs, yTrue):
    self.activation.forward(_inputs)
    self.output = self.activation.output
    return self.loss.calculate(self.output, yTrue)
  def backward(self, dValues, yTrue):
    samples = len(dValues)
    if len(yTrue.shape) == 2:
      yTrue = np.argmax(yTrue, axis=1)
    self.dInputs = dValues.copy()
    self.dInputs[range(samples), yTrue] -= 1 # calculate gradient
    self.dInputs = self.dInputs / samples # Normalize Gradient

class Main:
  nnfs.init()

  # nnfs book naming conventions for now
  X, y = spiral_data(samples=100, classes=3)

  # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
  # plt.show()

  layer1 = Layer_Dense(2,3)
  activation1 = Activation_ReLU()
  layer2 = Layer_Dense(3,3)
  lossActivation = ActivationSoftmaxLossCategoricalCrossEntropy()

  layer1.forward(X)
  activation1.forward(layer1.output)
  layer2.forward(activation1.output)

  loss = lossActivation.forward(layer2.output, y)

  print(lossActivation.output[:5])
  print('loss: ', loss)

  predictions = np.argmax(lossActivation.output, axis=1)
  if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
  accuracy = np.mean(predictions == y)

  print('acc: ', accuracy)

  lossActivation.backward(lossActivation.output, y)
  layer2.backward(lossActivation.dInputs)
  activation1.backward(layer2.dInputs)
  layer1.backward(activation1.dInputs)

  print("layer1 dWeights: ", layer1.dWeights)
  print("layer1 dBiases: ", layer1.dBiases)
  print("layer2 dWeights: ", layer2.dWeights)
  print("layer2 dBiases: ", layer2.dBiases)


if __name__ == "__main":
  main()



""" OUTPUT EXAMPLE(from my run)
  New set of weights found, epoch: 0 loss: 1.0988113 acc: 0.3333333333333333
  New set of weights found, epoch: 1 loss: 1.098686 acc: 0.3333333333333333
  New set of weights found, epoch: 9 loss: 1.0986263 acc: 0.3333333333333333
  New set of weights found, epoch: 42 loss: 1.0986255 acc: 0.3333333333333333
  New set of weights found, epoch: 108 loss: 1.0986218 acc: 0.3333333333333333
  New set of weights found, epoch: 185 loss: 1.0986147 acc: 0.3333333333333333
  New set of weights found, epoch: 281 loss: 1.0986133 acc: 0.3333333333333333
  New set of weights found, epoch: 759 loss: 1.0986124 acc: 0.09
"""
 