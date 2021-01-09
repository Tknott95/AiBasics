import numpy as np

netInputs = np.array([
  [0.88, 0.24, 0.14], [0.1, 0.76, 0.41], [0.02, 0.84, 0.08]])
classTargets  = np.array([[1, 0, 1]])

predictions = np.argmax(netInputs, axis=1)

# "nnfs note I am adding here from the book as this is funky to understand why I am doing this - if targets are one-hot encoded, convert them.
# Used this, prior to writing the code under this comment, right now: https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/
if len(classTargets.shape) == 2:
  classTargets = np.argmax(classTargets, axis=1)

accuracy = np.mean(predictions == classTargets)
print("Accuracy: ", accuracy)


# print("netInputs.shape: ", netInputs.shape)
