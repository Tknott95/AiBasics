from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

from collections import deque # Need to understand this pkg better - list w/ max size set .. etc
import time
import random
'''
  Not a fan of this abstraction pattern, tf2 docs have better ways, may be due to tf1? Will create my own in a diff way 
  Will do one w/ my own env + tf2.2 conventions

  Using tf-2.2 rather then ~tf-1.0 like tut
  Following Tutorial: https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
'''

REPLAY_MEM_SIZE = 50_000 # Normally hate caps yet will set like a c++ const as the tut does. Underscore works like a "comma"
MIN_REPLAY_MEM_SIZE = 1_000
MODEL_NAME = '256x2' # it is a 256x2 conv net
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

# keras wants to update log file every .fit so this will allow keras not to create a new log every .fit
# tf2.2 might not need this, looks like shit code
class ModifiedTensorBoard(TensorBoard):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.step = 1
    self.writer = tf.summary.FileWriter(self.log_dir)
  def set_model(self, model):
    pass
  def on_epoch_end(self, epoch, logs=None):
    self.update_stats(**logs)
  def on_batch_end(self, batch, logs=None):
    pass
  def on_train_end(self, _):
    pass
  def update_stats(self, **stats):
    self._write_logs(stats, self.step)

class DqlAgent:
  def __init__(self):
    self.model = self.createModel() # Gets trained every step
    
    self.targetModel = self.createModel() # Gets predicted against every step
    self.targetModel.set_weights(self.model.get_weights())

    self.replayMemory = deque(maxlen=REPLAY_MEM_SIZE)
    self.tensorboard = ModifiedTensorBoard(Log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
    
    self.targetUpdateCounter = 0


  def createModel(self):
    model = Sequential()
    model.add(Conv2D(256, (3,3), input_shape=envObservationSpaceVals))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(envActionSpaceSize, activation="linear"))
    model.compile(Loss="mse", optimizer=Adam(Lr=0.001), metrics=['accuracy'])
    return model

  def updateReplayMemory(self, transition):
    self.replayMemory.append(transition) # Allows for newQ form. In short transition is:  ObsSpace, action, reward.. new ObsSpace, isDone?

  def getQVals(self, terminalState, step):
    return self.modelPredict(np.array(state).reshape(-1, *state.shape)/255)[0] # /255 "scale" the rgb data getting passed in, as env is setup as such

  def train(self, terminalState, step):
    if len(self.replayMemory) < MIN_REPLAY_MEM_SIZE:
      return
    
    minibatch = random.sample(self.replayMemory, MINIBATCH_SIZE)
    currStates = np.array(transition[0] for transition in minibatch])/255 # /255 "scale" the rgb data getting passed in, as env is setup as such
    currQValsList = self.model.predict(currStates)

    newCurrStates = np.array([transition[3] for transition in minibatch])/255 # /255 "scale" the rgb data getting passed in, as env is setup as such
    futureQValsList = self.targetModel.predict(newCurrStates)

    X = []
    Y = []

    for index, (currState, action, reward, newCurrState, done) in enumerate(minibatch):
      if not done:
        maxFutureQVal = np.max(futureQValsList[index])
        newQVal = reward + DISCOUNT * maxFutureQVal
      else:
        newQVal = reward
    
      currQVals = currQValsList[index]
      currQVals[action] = newQVal

      X.append(currState)
      Y.append(currQVals)
    
    self.model.fit(np.array(X)/255, np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminalState else None)
    
    if terminalState:
      self.targetUpdateCounter += 1
    if self.targetUpdateCounter > UPDATE_TARGET_EVERY:
      self.targetModel.set_weights(self.model.get_weights())
      self.targetUpdateCounter = 0
