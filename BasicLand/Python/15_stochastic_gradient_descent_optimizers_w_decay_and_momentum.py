import numpy as np
# Pulling in nnfs data
import nnfs
from nnfs.datasets import spiral_data
# Loss is still acting funky will keep marching forward in book and then work backwards once solved

class LayerDense:
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


class ActivationReLU:
  def forward(self, _inputs):
    self._inputs = _inputs
    self.output = np.maximum(0, _inputs)
  def backward(self, dValues):
    self.dInputs = dValues.copy() 
    self.dInputs[self._inputs <= 0] = 0     # Zero gradient where input values were negative


class ActivationSoftmax:
  def forward(self, _inputs):
    self._inputs = _inputs
    exponential_values = np.exp(_inputs - np.max(_inputs, axis=1, keepdims=True))
    normalized_values = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
    self.output = normalized_values
  def backward(self, dValues):
    self.dInputs = np.empty_like(dValues) # unitialized array
    
    for index, (singleOutput, singleDValues) in \
                enumerate(zip(self.output, dValues)):
                singleOutput = singleOutput.reshape(-1, 1)
                jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T)
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
    self.activation = ActivationSoftmax()
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

class OptimizerSGD:
  def __init__(self, learningRate=1., decay=0., momentum=0.):
    self.learningRate = learningRate
    self.currLearningRate = learningRate
    self.decay = decay
    self.iterations = 0
    self.momentum = momentum
  def preUpdateParams(self):
    if self.decay:
      self.currLearningRate = self.learningRate * (1. / (1. + self.decay * self.iterations))
  def updateParams(self, layer):
    if self.momentum:
      # create mock 0 val momentum arrays if layer does not contain one
      if not hasattr(layer, 'weightMomentums'):
        layer.weightMomentums = np.zeros_like(layer.weights)
        layer.biasMomentums = np.zeros_like(layer.biases)
      
      weightUpdates = self.momentum * layer.weightMomentums - self.currLearningRate * layer.dWeights
      layer.weightMomentums = weightUpdates

      biasUpdates = self.momentum * layer.biasMomentums - self.currLearningRate * layer.dBiases
      layer.biasMomentums = biasUpdates
    else:
      weightUpdates = -self.currLearningRate * layer.dWeights
      biasUpdates = -self.currLearningRate * layer.dBiases
    
    layer.weights += weightUpdates
    layer.biases += biasUpdates
  def postUpdateParams(self):
    self.iterations += 1

class Main:
  nnfs.init()

  # nnfs book naming conventions for now
  X, y = spiral_data(samples=100, classes=3)


  layer1 = LayerDense(2,64)
  activation1 = ActivationReLU()
  layer2 = LayerDense(64,3)
  lossActivation = ActivationSoftmaxLossCategoricalCrossEntropy()

  optimizer = OptimizerSGD(decay=1e-3, momentum=0.5)
 
  for epoch in range(10044):
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)

    loss = lossActivation.forward(layer2.output, y)

    predictions = np.argmax(lossActivation.output, axis=1)
    if len(y.shape) == 2:
      y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
      print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f}, ' +
            f'lr: {optimizer.currLearningRate}')

    lossActivation.backward(lossActivation.output, y)
    layer2.backward(lossActivation.dInputs)
    activation1.backward(layer2.dInputs)
    layer1.backward(activation1.dInputs)

    optimizer.preUpdateParams()
    optimizer.updateParams(layer1)
    optimizer.updateParams(layer2)
    optimizer.postUpdateParams()

    #  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
    #  plt.show()


if __name__ == "__main":
  main()
