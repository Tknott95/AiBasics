import numpy as np
# Pulling in nnfs data
import nnfs
from nnfs.datasets import sine_data
# Not using a dropout layer as it killed accuracy

class LayerDense:
    def __init__(self, _numOfInputs, _numOfNeurons, 
    weightRegularizerL1=0, weightRegularizerL2=0, biasRegularizerL1=0, biasRegularizerL2=0):
      self.weights = 0.1 * np.random.randn(_numOfInputs, _numOfNeurons)
      self.biases = np.zeros((1, _numOfNeurons))

      self.weightRegularizerL1 = weightRegularizerL1
      self.weightRegularizerL2 = weightRegularizerL2
      self.biasRegularizerL1 = biasRegularizerL1
      self.biasRegularizerL2 = biasRegularizerL2
    def forward(self, _inputs):
      self._inputs = _inputs
      self.output = np.dot(_inputs, self.weights) + self.biases
    def backward(self, dValues): # dValues is derivativeValues
      # My gradients on params
      self.dWeights = np.dot(self._inputs.T, dValues)
      self.dBiases = np.sum(dValues, axis=0, keepdims=True)

      if self.weightRegularizerL1 > 0:
        dRegL1 = np.ones_like(self.weights)
        dRegL1[self.weights < 0] = -1
        self.dWeights += self.weightRegularizerL1 * dRegL1
      if self.weightRegularizerL2 > 0:
        self.dWeights += 2 * self.weightRegularizerL2 * self.weights
      
      if self.biasRegularizerL1 > 0:
        dRegL1 = np.ones_like(self.biases)
        dRegL1[self.biases < 0] = -1
        self.dBiases += self.biasRegularizerL1 * dRegL1
      if self.biasRegularizerL2 > 0:
        self.dBiases += 2 * self.biasRegularizerL2 * self.biases

      # My gradient on values
      self.dInputs = np.dot(dValues, self.weights.T)
  
class LayerDropout:
  def __init__(self, rate):
    self.rate = 1 - rate
  def forward(self, inputs):
    self.inputs = inputs
    self.binaryMask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
    self.output = inputs * self.binaryMask
  def backward(self, dValues):
    self.dInputs = dValues * self.binaryMask

class ActivationLinear:
  def forward(self, _inputs):
    self._inputs = _inputs
    self.output = _inputs
  def backward(self, dValues):
    self.dInputs = dValues.copy()

class ActivationReLU:
  def forward(self, _inputs):
    self._inputs = _inputs
    self.output = np.maximum(0, _inputs)
  def backward(self, dValues):
    self.dInputs = dValues.copy() 
    self.dInputs[self._inputs <= 0] = 0 # Zero gradient where input values were negative

class ActivationSoftmax:
  def forward(self, _inputs):
    self._inputs = _inputs
    exponential_values = np.exp(_inputs - np.max(_inputs, axis=1, keepdims=True))
    normalized_values = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
    self.output = normalized_values
  def backward(self, dValues):
    self.dInputs = np.empty_like(dValues)
    
    for index, (singleOutput, singleDValues) in enumerate(zip(self.output, dValues)):
      singleOutput = singleOutput.reshape(-1, 1)
      jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T)
      self.dInputs[index] = np.dot(jacobianMatrix, singleDValues) 
class ActivationSigmoid:
  def forward(self, _inputs):
    self._inputs = _inputs
    self.output = 1 / (1 + np.exp(-_inputs))
  def backward(self, dValues):
    self.dInputs = dValues * (1 - self.output) * self.output

class Loss:
  def regularizationLoss(self, layer):
    regularizationLoss = 0
    
    if layer.weightRegularizerL1 > 0:
      regularizationLoss += layer.weightRegularizerL1 * np.sum(np.abs(layer.weights))
    if layer.weightRegularizerL2 > 0:
      regularizationLoss += layer.weightRegularizerL2 * np.sum(layer.weights**2)
    if layer.biasRegularizerL1 > 0:
      regularizationLoss += layer.biasRegularizerL1 * np.sum(np.abs(layer.biases))
    if layer.biasRegularizerL2 > 0:
      regularizationLoss += layer.biasRegularizerL2 * np.sum(layer.biases**2)

    return regularizationLoss
  def calculate(self, output, y):
    sampleLosses = self.forward(output, y)
    dataLoss = np.mean(sampleLosses)

    return dataLoss

class CategoricalCrossEntropyLoss(Loss):
  def forward(self, yPrediction, yTrue):
    samples = len(yPrediction)
    """ NNFS note on line/code below
      Clip data to prevent division by 0. 
      Clip both sides to not drag mean towards any value """
    yPredictionClipped = np.clip(yPrediction, 1e-7, 1 - 1e-7)

    if len(yTrue.shape) == 1:
      correctConfidences = yPredictionClipped[
        range(samples),
        yTrue
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
class BinaryCrossEntropyLoss(Loss):
  def forward(self, yPrediction, yTrue):
    yPredictionClipped = np.clip(yPrediction, 1e-7, 1 - 1e-7)
    sampleLosses = -(yTrue * np.log(yPredictionClipped) + (1 - yTrue) * np.log(1 - yPredictionClipped))
    sampleLosses = np.mean(sampleLosses, axis=1)
    
    return sampleLosses
  def backward(self, dValues, yTrue):
    samples = len(dValues)
    outputs = len(dValues[0])

    clippedDValues = np.clip(dValues, 1e-7, 1 - 1e-7)
    self.dInputs = -(yTrue / clippedDValues - (1 - yTrue) / (1 - clippedDValues)) / outputs
    self.dInputs = self.dInputs / samples
class MeanSquaredErrorLoss(Loss):
  def forward(self, yPred, yTrue):
    sampleLosses = np.mean((yTrue - yPred)**2, axis=-1) # HAD axis=1 instead of -1
    return sampleLosses
  def backward(self, dValues, yTrue):
    samples = len(dValues)
    outputs = len(dValues[0])
    self.dInputs = -2 * (yTrue - dValues) / outputs
    self.dInputs = self.dInputs / samples
class MeanAbsoluteErrorLoss(Loss):
  def forward(self, yPred, yTrue):
    sampleLosses = np.mean((yTrue - yPred), axis=-1) # HAD axis=1 instead of -1
    return sampleLosses
  def backward(self, dValues, yTrue):
    samples = len(dValues)
    outputs = len(dValues[0])
    self.dInputs = np.sign(yTrue - dValues) / outputs # np.sign -> -1 if x < 0, 0 if x==0, 1 if x > 0.
    self.dInputs = self.dInputs / samples

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

# BRING IN OTHER OPTIMIZERS LATER @TODO

class OptimizerAdam: # Adam -> Adaptive Momentum
  def __init__(self, learningRate=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
    self.learningRate = learningRate
    self.currLearningRate = learningRate
    self.decay = decay
    self.iterations = 0
    self.epsilon = epsilon
    self.beta1 = beta1
    self.beta2 = beta2
  def preUpdateParams(self):
    if self.decay:
      self.currLearningRate = self.learningRate * (1. / (1. + self.decay * self.iterations))
  def updateParams(self, layer):
    # create mock 0 val momentum arrays if layer does not contain one
    if not hasattr(layer, 'weightCache'):
      layer.weightMomentums = np.zeros_like(layer.weights)
      layer.weightCache = np.zeros_like(layer.weights)
      layer.biasMomentums = np.zeros_like(layer.biases)
      layer.biasCache = np.zeros_like(layer.biases)
    
    layer.weightMomentums = self.beta1 * layer.weightMomentums + (1 - self.beta1) * layer.dWeights
    layer.biasMomentums = self.beta1 * layer.biasMomentums + (1 + self.beta1) * layer.dBiases

    weightMomentumsCorrected = layer.weightMomentums / (1 - self.beta1 ** (self.iterations + 1))
    biasMomentumsCorrected = layer.biasMomentums / (1 - self.beta1 ** (self.iterations + 1))
    
    layer.weightCache = self.beta2 * layer.weightCache + (1 - self.beta2) * layer.dWeights**2
    layer.biasCache = self.beta2 * layer.biasCache + (1 - self.beta2) * layer.dBiases**2

    weightCacheCorrected = layer.weightCache / (1 - self.beta2 ** (self.iterations + 1))
    biasCacheCorrected = layer.biasCache / (1 - self.beta2 ** (self.iterations + 1))

    layer.weights += -self.currLearningRate * weightMomentumsCorrected / (np.sqrt(weightCacheCorrected) + self.epsilon)
    layer.biases += -self.currLearningRate * biasMomentumsCorrected / (np.sqrt(biasCacheCorrected) + self.epsilon)

  def postUpdateParams(self):
    self.iterations += 1

class Main:
  nnfs.init()

  X, y = sine_data()

  denseLayer1 = LayerDense(1, 64) # weightRegularizerL2=5e-4, biasRegularizerL2=5e-4
  activation1 = ActivationReLU()
  # dropoutLayer1 = LayerDropout(0.1)
  activation2 = ActivationReLU()
  denseLayer2 = LayerDense(64, 64)
  activation3 = ActivationLinear()
  denseLayer3 = LayerDense(64, 1)
  lossFunction = MeanSquaredErrorLoss()

  optimizer = OptimizerAdam(learningRate=0.003, decay=1e-3)
  accuracyPrecision = np.std(y) / 250
 
  for epoch in range(9844):
    denseLayer1.forward(X)
    activation1.forward(denseLayer1.output)
    # dropoutLayer1.forward(activation1.output)
    denseLayer2.forward(activation1.output)
    activation2.forward(denseLayer2.output)

    denseLayer3.forward(activation2.output)
    activation3.forward(denseLayer3.output)

    dataLoss = lossFunction.calculate(activation3.output, y)
    regularizationLoss = lossFunction.regularizationLoss(denseLayer1) + lossFunction.regularizationLoss(denseLayer2) + lossFunction.regularizationLoss(denseLayer3)
    loss = dataLoss + regularizationLoss

    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracyPrecision)

    if not epoch % 100:
      print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f}, ' +
            f'dataLoss: {dataLoss:.3f}, ' +
            f'regLoss: {regularizationLoss:.3f}, ' +
            f'lr: {optimizer.currLearningRate:.5}')

    lossFunction.backward(activation3.output, y)
    activation3.backward(lossFunction.dInputs)
    denseLayer3.backward(activation3.dInputs)
    activation2.backward(denseLayer3.dInputs)
    denseLayer2.backward(activation2.dInputs)
    # dropoutLayer1.backward(denseLayer2.dInputs)
    activation1.backward(denseLayer2.dInputs)
    denseLayer1.backward(activation1.dInputs)

    optimizer.preUpdateParams()
    optimizer.updateParams(denseLayer1)
    optimizer.updateParams(denseLayer2)
    optimizer.updateParams(denseLayer3)
    optimizer.postUpdateParams()
  
  # TESTING AND PLOTTING, NO VALIDATION 
  import matplotlib.pyplot as plt
  xTest, ytest = sine_data()

  denseLayer1.forward(xTest)
  activation1.forward(denseLayer1.output)
  denseLayer2.forward(activation1.output)
  activation2.forward(denseLayer2.output)
  denseLayer3.forward(activation2.output)
  activation3.forward(denseLayer3.output)

  plt.plot(xTest, ytest)
  plt.plot(xTest, activation3.output)
  plt.show()


if __name__ == "__main":
  main()
