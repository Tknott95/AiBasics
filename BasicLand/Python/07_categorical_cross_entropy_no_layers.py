import numpy as np

# As said in the read me this code is very heavily nnfs book w/ twkz to learn n.n.f.s
# netOutputs = np.array([
#   [0.88, 0.24, 0.14], [0.1, 0.76, 0.41], [0.02, 0.84, 0.08]])
# targetOutputs = np.array([
#   [1, 0, 0], [0, 1, 0], [0, 1, 0]])

netOutputs = np.array([[0.88, 0.24, 0.14]])
targetOutputs = np.array([[1, 0, 0]])

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
        yTrue
      ]
    elif len(yTrue.shape) == 2:
      correctConfidences = np.sum(
        yPredictionClipped*yTrue,
        axis=1
      )
    
    negativeLogLikelihoods = -np.log(correctConfidences)
    return negativeLogLikelihoods

lossFunction = CategoricalCrossEntropyLoss()
lossVal = lossFunction.calculate(netOutputs, targetOutputs)
print("  Loss:  ", lossVal)