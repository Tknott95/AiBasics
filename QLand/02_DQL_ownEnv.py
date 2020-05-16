from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np

from collections import deque # list w/ max size set .. etc
from PIL import Image
import cv2
import time
import random

from datetime import datetime
from packaging import version
import os

'''
  Not a fan of this abstraction pattern, tf2 docs have better ways, may be due to tf1? Will create my own in a diff way 
  Will do one w/ my own env + tf2.2 conventions

  Using tf-2.2 rather then ~tf-1.0 like tut
  Following Tutorial: https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
'''

# device_name = tf.test.gpu_device_name()
# if not device_name:
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))


replayMemSize = 50_000 # Underscore works like a "comma"
minReplayMemSize = 1_000
modelName = '256x2' # it is a 256x2 conv net
miniBatchSize = 64
agentagentDiscount = 0.99
updateTargetEvery = 5

class Blob:
  def __init__(self, size):
    self.size = size
    self.x = np.random.randint(0, size)
    self.y = np.random.randint(0, size)
  def __str__(self):
    return f"Blob ({self.x}, {self.y})"
  def __sub__(self, other):
    return (self.x - other.x, self.y - other.y)
    
  def action(self, choice):
    if choice == 0:
      self.move(x=1, y=1)
    elif choice == 1:
      self.move(x=-1, y=-1)
    elif choice == 2:
      self.move(x=-1, y=1)
    elif choice == 3:
      self.move(x=1, y=-1)
    elif choice == 4:
      self.move(x=1, y=0)
    elif choice == 5:
      self.move(x=-1, y=0)
    elif choice == 6:
      self.move(x=0, y=1)
    elif choice == 7:
      self.move(x=0, y=-1)
    elif choice == 8:
      self.move(x=0, y=0)

  def move(self, x=False, y=False):
    if not x:
      self.x += np.random.randint(-1, 2)
    else:
      self.x += x
    if not y:
      self.y += np.random.randint(-1, 2)
    else:
      self.y += y

    if self.x < 0: # Boundaries, clearly
      self.x = 0
    elif self.x > self.size-1:
      self.x = self.size-1
    if self.y < 0:
      self.y = 0
    elif self.y > self.size-1:
      self.y = self.size-1

class BlobEnv:
  gridSize = 10
  returnImages = True
  movePenalty = 1
  enemyPenalty = 300
  foodReward = 25
  obsSpaceVals = (gridSize, gridSize, 3)
  actionSpaceSize = 9
  playerN = 1
  foodN = 2
  enemyN = 3
  dictionary = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)} # BGR format

  def reset(self):
    self.player = Blob(self.gridSize)
    self.food = Blob(self.gridSize)
    while self.food == self.player:
      self.food = Blob(self.gridSize)
      self.enemy = Blob(self.gridSize)
    while self.enemy == self.player or self.enemy == self.food:
      self.enemy = Blob(self.gridSize)
      
    self.episodeStep = 0

    if self.returnImages:
      observation = np.array(self.get_image())
    else:
      observation = (self.player-self.food) + (self.player-self.enemy)
    return observation

  def step(self, action):
    self.episodeStep += 1
    self.player.action(action)

    if self.returnImages:
      newObs = np.array(self.get_image())
    else:
      newObs = (self.player-self.food) + (self.player-self.enemy)

    if self.player == self.enemy:
      reward = -self.enemyPenalty
    elif self.player == self.food:
      reward = self.foodReward
    else:
      reward = -self.movePenalty

    done = False
    if reward == self.foodReward or reward == -self.enemyPenalty or self.episodeStep >= 200:
      done = True

    return newObs, reward, done

  def render(self):
    img = self.get_image()
    img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
    cv2.imshow("image", np.array(img))  # show it!
    cv2.waitKey(1)

  def get_image(self):
    env = np.zeros((self.gridSize, self.gridSize, 3), dtype=np.uint8)  # rbg of gridSize
    env[self.food.x][self.food.y] = self.dictionaryictionary[self.foodN]  # food loc tile to green color
    env[self.enemy.x][self.enemy.y] = self.dictionaryictionary[self.enemyN]  # enemy loc tile to red
    env[self.player.x][self.player.y] = self.dictionaryictionary[self.playerN]  # player loc tile to blue
    img = Image.fromarray(env, 'RGB') 
    return img


env = BlobEnv()
epRewards = [-200]

random.seed(1) # Seed for same results, seeding is basic bish
np.random.seed(1)
tf.random.set_seed(1) # Chaned from set_random_seed -> random.set_seed in tf2.0

if not os.path.isdir('models'):
  os.makedirs('models')

# keras wants to update log file every .fit so this will allow keras not to create a new log every .fit
# tf2.2 might not need this, looks like shit code
class ModifiedTensorBoard(TensorBoard):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.step = 1
    self.writer = tf.summary.create_file_writer(self.log_dir)
  def set_model(self, model):
    pass
  def on_epoch_end(self, epoch, logs=None):
    self.update_stats(**logs)
  def on_batch_end(self, batch, logs=None):
    pass
  def on_train_end(self, _):
    pass
  def update_stats(self, **stats):
    tf.summary.scalar('loss',stats['loss'], step=self.step)

class DQLAgent:
  def __init__(self):
    self.model = self.createModel() # Gets trained every step
    
    self.targetModel = self.createModel() # Gets predicted against every step
    self.targetModel.set_weights(self.model.get_weights())

    self.replayMemory = deque(maxlen=replayMemSize)
    self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{modelName}-{int(time.time())}")
    
    self.targetUpdateCounter = 0


  def createModel(self):
    model = Sequential()
    model.add(Conv2D(256, (3,3), input_shape=env.obsSpaceVals))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(env.actionSpaceSize, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

  def updateReplayMemory(self, transition):
    self.replayMemory.append(transition) # Allows for newQ form. In short transition is:  ObsSpace, action, reward.. new ObsSpace, isDone?

  def getQVals(self, terminalState, step):
    return self.modelPredict(np.array(state).reshape(-1, *state.shape)/255)[0] # /255 "scale" the rgb data getting passed in, as env is setup as such

  def train(self, terminalState, step):
    if len(self.replayMemory) < minReplayMemSize:
      return
    
    minibatch = random.sample(self.replayMemory, miniBatchSize)
    currStates = np.array([transition[0] for transition in minibatch])/255 # /255 "scale" the rgb data getting passed in, as env is setup as such
    currQValsList = self.model.predict(currStates)

    newCurrStates = np.array([transition[3] for transition in minibatch])/255 # /255 "scale" the rgb data getting passed in, as env is setup as such
    futureQValsList = self.targetModel.predict(newCurrStates)

    X = []
    Y = []

    for index, (currState, action, reward, newCurrState, done) in enumerate(minibatch):
      if not done:
        maxFutureQVal = np.max(futureQValsList[index])
        newQVal = reward + agentagentDiscount * maxFutureQVal
      else:
        newQVal = reward
    
      currQVals = currQValsList[index]
      currQVals[action] = newQVal

      X.append(currState)
      Y.append(currQVals)
    
    self.model.fit(np.array(X)/255, np.array(Y), batch_size=miniBatchSize, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminalState else None)
    
    if terminalState:
      self.targetUpdateCounter += 1
    if self.targetUpdateCounter > updateTargetEvery:
      self.targetModel.set_weights(self.model.get_weights())
      self.targetUpdateCounter = 0

myAgent = DQLAgent()
