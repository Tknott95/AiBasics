import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time

''' 
  Following "Sentdex" example/teaching w/ own tweaks/additions, etc.
  TutLink: https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/   
'''
def main():
  style.use("ggplot")

  SIZE = 10
  agentEpisodes = 24000
  movePenalty = 1
  enemyPenalty = 300
  foodReward = 25

  epsilon = 0.9
  epsilonDecay = 0.9998
  showEveryEps = 2000

  startQTable = None

  learningRate = 0.1
  discount = 0.95

  playerN = 1
  foodN = 2
  enemyN = 3

  d = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)} # BGR format


  class Blob:
    def __init__(self):
      self.x = np.random.randint(0, SIZE)
      self.y = np.random.randint(0, SIZE)
    def __str__(self):
      return f"{self.x}, {self.y}"
    def __sub__(self, other):
      return (self.x - other.x, self.y - other.y)
    
    def action(self):
      if choice == 0:
        self.move(x=1, y=1)
      elif choice == 1:
        self.move(x=-1, y=-1)
      elif choice == 2:
        self.move(x=-1, y=1)
      elif choice == 3:
        self.move(x=1, y=-1)

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
      elif self.x > SIZE-1:
        self.x = SIZE-1
      if self.y < 0:
        self.y = 0
      elif self.y > SIZE-1:
        self.y = SIZE-1

  if startQTable is None: # Creating qTable if ! have one
    qTable = {}
    # Every combo
    for x1 in range(-SIZE+1, SIZE):
      for y1 in range(-SIZE+1, SIZE):
        for x2 in range(-SIZE+1, SIZE):
          for y2 in range(-SIZE+1, SIZE):
            qTable[((x1, y1), (x2, y2))] = [np.random.uniform(-5,0) for i in range(4)]
  else:
    with open(startQTable, "rb") as f: # Loading pretrained file if have one.
      qTable = pickle.load(f)

  episodeRewards = []
  for episode in range(agentEpisodes):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % showEveryEps == 0:
      print(f"on # {episode}, epsilon: {epsilon}")
      print(f"{showEveryEps} ep mean {np.mean(episodeRewards[-showEveryEps:])}")
      show = True
    else:
      show = False
    
    episodeReward = 0
    for i in range(200): #stepsToTake is 200
      observation = (player-food, player-enemy)
      if np.random.random() > epsilon:
        action = np.argmax(qTable[observation])
      else:
        action = np.random.randint(0, 4)      
      player.action(action)

      if player.x == enemy.x and player.y == enemy.y:
        reward = -enemyPenalty
      elif player.x == food.x and player.y == food.y:
        reward = foodReward
      else:
        reward = -movePenalty
      
      newObservation = (player-food, player-enemy)
      maxFutureQ = np.max(qTable[newObservation])
      currentQ = qTable[observation][action]

      if reward == foodReward:
        newQ = foodReward
      elif reward == -enemyPenalty
        newQ = -enemyPenalty
      else:
        newQ = (1-learningRate) * currentQ + learningRate * (reward + discount * maxFutureQ) # Actual qLearning Algo
      
      qTable[observation][action] = newQ




if __name__ == "__main__":
  main()