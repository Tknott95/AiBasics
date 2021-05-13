import numpy as np
# Pulling in nnfs data
import nnfs

import os
import urllib
import urllib.request

from zipfile import ZipFile

import cv2

# Libs for pulling my zip and decompressing it
DATA_URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
DATA_FILE = 'fashion_mnist_images.zip'
DATA_FOLDER = 'Assets/fashion_mnist_images'

"""
  @NOTE : BRING IN THE VALIDATION DATA INTO MY MODEL.train() FOR INLINE TESTING ...
  ... RATHER THAN MY WACKY ASS OLD HACK

  @TODO : 
    - PULL IN LIB TO HANDLE A PROGRESS BAR IN THE SHELL
    - IF LAZY JUST LOG EVERY ITERATION
"""

"""
      print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f}, ' +
            f'dataLoss: {dataLoss:.3f}, ' +
            f'regLoss: {regularizationLoss:.3f}, ' +
            f'lr: {optimizer.currLearningRate:.5}')
"""
# @TODO clean up var/param names, pulled this in fro mmy u24 asset pull script
def loadMnistData(data, path):
  if not os.path.isfile(DATA_FILE):
    print(f'\n  Downloading {DATA_URL} and saving as {DATA_FILE}...')
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
  
    print('\n  Decompressing Images...')
    with ZipFile(DATA_FILE) as zipImages:
      zipImages.extractall(DATA_FOLDER)

  labels = os.listdir(os.path.join(path, data))
  # X, y = [] throws a size error of int 0 instead of 2 so we jsut set each individually
  X = []
  y = []
  
  for label in  labels:
    for file in os.listdir(os.path.join(path, data, label)):
      image = cv2.imread(os.path.join(path, data, label, file), cv2.IMREAD_UNCHANGED)
      X.append(image)
      y.append(label)
  
  return np.array(X), np.array(y).astype('uint8')

def createMnistData(path):
  X, y = loadMnistData('train', path)
  xTest, yTest = loadMnistData('test', path)

  return X, y, xTest, yTest


class LayerDense:
    def __init__(self, _numOfInputs, _numOfNeurons, weightRegularizerL1=0, weightRegularizerL2=0, biasRegularizerL1=0, biasRegularizerL2=0):
      self.weights = 0.1 * np.random.randn(_numOfInputs, _numOfNeurons)
      self.biases = np.zeros((1, _numOfNeurons))

      self.weightRegularizerL1 = weightRegularizerL1
      self.weightRegularizerL2 = weightRegularizerL2
      self.biasRegularizerL1 = biasRegularizerL1
      self.biasRegularizerL2 = biasRegularizerL2

    def forward(self, _inputs, isTraining):
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
  def forward(self, _inputs, isTraining):
    self._inputs = _inputs

    if not isTraining:
      self.output = inputs.copy()
      return
  
    self.binaryMask = np.random.binomial(1, self.rate, size=_inputs.shape) / self.rate
    self.output = _inputs * self.binaryMask
  def backward(self, dValues):
    self.dInputs = dValues * self.binaryMask

class LayerInput():
  def forward(self, _inputs, isTraining):
    self.output = _inputs

class ActivationLinear:
  def forward(self, _inputs, isTraining):
    self._inputs = _inputs
    self.output = _inputs
  def backward(self, dValues):
    self.dInputs = dValues.copy()

class ActivationReLU:
  def forward(self, _inputs, isTraining):
    self._inputs = _inputs
    self.output = np.maximum(0, _inputs)
  def backward(self, dValues):
    self.dInputs = dValues.copy() 
    self.dInputs[self._inputs <= 0] = 0 # Zero gradient where input values were negative

class ActivationSoftmax:
  def forward(self, _inputs, isTraining):
    self._inputs = _inputs
    exponential_values = np.exp(_inputs - np.max(_inputs, axis=1, keepdims=True))
    normalized_values = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
    self.output = normalized_values
  def backward(self, dValues): # Double check dis
    self.dInputs = np.empty_like(dValues)
    
    for index, (singleOutput, singleDValues) in enumerate(zip(self.output, dValues)):
      singleOutput = singleOutput.reshape(-1, 1)
      jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T)
      self.dInputs[index] = np.dot(jacobianMatrix, singleDValues)
  def predictions(self, outputs):
    return np.argmax(outputs, axis=1)

class ActivationSigmoid:
  def forward(self, _inputs):
    self._inputs = _inputs
    self.output = 1 / (1 + np.exp(-_inputs))
  def backward(self, dValues):
    self.dInputs = dValues * (1 - self.output) * self.output

class Loss:
  def regularizationLoss(self):
    regularizationLoss = 0
    
    for layer in self.trainableLayers:
      if layer.weightRegularizerL1 > 0:
        regularizationLoss += layer.weightRegularizerL1 * np.sum(np.abs(layer.weights))
      if layer.weightRegularizerL2 > 0:
        regularizationLoss += layer.weightRegularizerL2 * np.sum(layer.weights**2)
      if layer.biasRegularizerL1 > 0:
        regularizationLoss += layer.biasRegularizerL1 * np.sum(np.abs(layer.biases))
      if layer.biasRegularizerL2 > 0:
        regularizationLoss += layer.biasRegularizerL2 * np.sum(layer.biases**2)

    return regularizationLoss
  def persistTrainableLayers(self, trainableLayers):
    self.trainableLayers = trainableLayers
  def calculate(self, output, y, *, includeRegularization=False):
    sampleLosses = self.forward(output, y)
    dataLoss = np.mean(sampleLosses)

    self.accumulatedSum += np.sum(sampleLosses)
    self.accumulatedCount += len(sampleLosses)    

    if not includeRegularization:
      return dataLoss

    return dataLoss, self.regularizationLoss()
  def calculateAccumulated(self, *, includeRegularization=False):
    dataLoss = self.accumulatedSum / self.accumulatedCount
    
    if not includeRegularization:
      return dataLoss

    return dataLoss, self.regularizationLoss()
  def newPass(self):
    self.accumulatedSum = 0
    self.accumulatedCount = 0

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

# @TODO BRING IN OTHER OPTIMIZERS LATER
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


class Accuracy:
  def calculate(self, predictions, y):
    comparisons = self.compare(predictions, y)
    accuracy = np.mean(comparisons)

    self.accumulatedSum += np.sum(comparisons)
    self.accumulatedCount += len(comparisons)

    return accuracy
  def calculateAccumulated(self):
    accuracy = self.accumulatedSum / self.accumulatedCount
    return accuracy
  def newPass(self):
    self.accumulatedSum = 0
    self.accumulatedCount = 0
 

class CategoricalAccuracy(Accuracy):
  def __init__(self, *, binary=False):
    self.binary = binary
  def init(self, y):
    pass
  def compare(self, predictions, y):
    if not self.binary and len(y.shape) == 2:
      y = np.argmax(y, axis=1)
    return predictions == y
  

class Model:
  # @NOTE - ACCURACY WAS PULLED UNTIL CPLX MVP 0001
  def __init__(self):
    self.layers = []
    self.softmaxClassifierOutput = None
  def add(self, layer):
    self.layers.append(layer)
  def set(self, *, loss, optimizer, accuracy): #, accuracy
    self.loss = loss
    self.optimizer = optimizer
    self.accuracy = accuracy
  def finalize(self):
    self.inputLayer = LayerInput()
    layerCount = len(self.layers)
    print('layerCount: ', layerCount)
    self.trainableLayers = []
  
    for j in range(layerCount):
      if j == 0:
        self.layers[j].prev = self.inputLayer
        self.layers[j].next = self.layers[j+1]
      elif j < layerCount - 1:
        self.layers[j].prev = self.layers[j-1]
        self.layers[j].next = self.layers[j+1]
      else:
        self.layers[j].prev = self.layers[j-1]
        self.layers[j].next = self.loss
        self.outputLayerActivation = self.layers[j]
    
      if hasattr(self.layers[j], 'weights'):
        self.trainableLayers.append(self.layers[j])
    
    self.loss.persistTrainableLayers(self.trainableLayers)
    if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(self.loss, CategoricalCrossEntropyLoss):
      self.softmaxClassifierOutput = ActivationSoftmaxLossCategoricalCrossEntropy()
    print('after layerCount: ', layerCount)

  def train(self, x, y, *, batchSize=None, epochs=1000, logEvery=10, validationData=None):
    # @NOTE when accuracy is put in add:
    self.accuracy.init(y)
    # @NOTE if no batchSize trainSteps = 1, @TODO change this later
    trainSteps = 1

    if validationData is not None:
      validationSteps = 1
      xVal, yVal = validationData
   
    if batchSize is not None:
      trainSteps = len(x)
      if trainSteps * batchSize < len(x):
        trainSteps += 1 # @TODO make ++ ? 
      if validationData is not None:  
        validationSteps = len(xVal)
        
        if validationSteps * batchSize < len(xVal): validationSteps += 1
     
    for epoch in range(1, epochs+1):
      self.loss.newPass()
      self.accuracy.newPass()
      
      for step in range(trainSteps):
        if batchSize is None:
          batchX = x
          batchY = y
        else:
          batchX = x[step*batchSize:(step+1)*batchSize]
          batchY = y[step*batchSize:(step+1)*batchSize]


        output = self.forward(batchX, isTraining=True)
        # print(epoch)
        # @TODO Print my actual epoch data via. each train
        dataLoss, regularizationLoss = self.loss.calculate(output, batchY, includeRegularization=True)
        # @TODO FIX THIS -> loss = dataLoss + regularizationLoss
        loss = dataLoss + regularizationLoss

        predictions = self.outputLayerActivation.predictions(output)
        accuracy = self.accuracy.calculate(predictions, batchY)
      

        self.backward(output, batchY)

        self.optimizer.preUpdateParams()
        for layer in self.trainableLayers:
         self.optimizer.updateParams(layer)
        self.optimizer.postUpdateParams()

        if not step % logEvery or step == trainSteps-1:
          print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'dataLoss: {dataLoss:.3f}, ' +
              f'regLoss: {regularizationLoss}, ' +
              f'lr: {self.optimizer.currLearningRate:.5}')
        # @TODO fix regularizartionLoss # 529
    
    if validationData is not None:
      xVal, yVal = validationData
      output = self.forward(xVal, isTraining = False)
      loss = self.loss.calculate(output, yVal)
      predictions = self.outputLayerActivation.predictions(output)
      accuracy = self.accuracy.calculate(predictions, yVal)
      print(f'validation, ' +
            f'acc: {accuracy:.3f} , ' +
            f'loss: {loss:.3f}')

  def forward(self, x, isTraining):
    self.inputLayer.forward(x, isTraining)
    for layer in self.layers:
      layer.forward(layer.prev.output, isTraining)
    
    return layer.output
  
  def backward(self, output, y):
    if self.softmaxClassifierOutput is not None:
      self.softmaxClassifierOutput.backward(output, y)

      self.layers[-1].dInputs = self.softmaxClassifierOutput.dInputs

      for layer in reversed(self.layers[:-1]):
        layer.backward(layer.next.dInputs)
      
      return
    
    self.loss.backward(output, y)

    for layer in reversed(self.layers):
      layer.backward(layer.next.dInputs)
    

class Main:
  nnfs.init()

  x, y, xTest, yTest = createMnistData('Assets/fashion_mnist_images')

  keys = np.array(range(x.shape[0])) # shuffling the training dataset
  np.random.shuffle(keys)
  x = x[keys]
  y = y[keys]

  x = (x.reshape(x.shape[0], -1).astype(np.float32) - 127.5) / 127.5 # scaling and reshaping samples
  xTest = (xTest.reshape(xTest.shape[0], -1).astype(np.float32) - 127.5) / 127.5

  epochs = 444
  batchSize = 128
 
  model = Model()

  model.add(LayerDense(x.shape[1], 128, weightRegularizerL2=5e-4, biasRegularizerL2=5e-4))
  model.add(ActivationReLU())
  # model.add(LayerDropout(0.1)) @NOTE bring back in l8r
  model.add(LayerDense(128, 128))
  model.add(ActivationReLU())
  model.add(LayerDense(128, 10))
  model.add(ActivationSoftmax())

  layerCount = len(model.layers) # move into model and call self @TODO
  print("\nlayerCount: ", layerCount)
  print("epochs: \n  ", epochs)
  print("layers: \n  ", model.layers)

  # @TODO bring in accuracy as a parameter

  # Was MeanSquaredErrorLoss()
  model.set(loss=CategoricalCrossEntropyLoss(), optimizer=OptimizerAdam(learningRate=5e-3, decay=1e-3), accuracy=CategoricalAccuracy())
  model.finalize()
  model.train(x, y, validationData=(xTest, yTest),  epochs=epochs, logEvery=1)


if __name__ == "__main":
  main()
