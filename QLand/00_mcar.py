import gym
import numpy as np

def main():
  ''' Following "Sentdex" example/teaching w/ own tweaks/additions, etc. '''
  env = gym.make("MountainCar-v0")

  agentLearningRate = 0.1 # Can be between 0-1
  agentDiscount = 0.95    # Essentially like a "weight"
  agentEpisodes = 25000   # Essentially like "epochs"

  print('\n env.action_space: ', env.action_space)
  print(' env.observation_space: ', env.observation_space)
  print(' env.observation_space.high: ', env.observation_space.high)
  print(' env.observation_space.low: ', env.observation_space.low, '\n')

  discreteObservationSize = [20] * len(env.observation_space.high) # Should not be hardcoded
  discreteObservationWinSize = (
    env.observation_space.high - env.observation_space.low) / discreteObservationSize

  print('This is not dynamic as it should be, just MVP')
  print(' discreteObservationSize: ',  discreteObservationSize)
  print(' discreteObservationWinSize: ', discreteObservationWinSize)

  ''' 
    Init() the qTable as a [20]x[20]x[3]
    This is the size(discreteObservationSize + [env.action_space.n]) 
    (discreteObservationSize & every action possible[3] so (20, 20, 3) -> EveryObservationCombo)
  '''
  qTable = np.random.uniform(low=-2,high=0,size=(discreteObservationSize + [env.action_space.n]))
  print('\n qTable.shape: ', qTable.shape)
  # print('qTable\n', qTable, '\n')

  discreteState = getDiscreteState(env.reset(), env.observation_space.low, discreteObservationWinSize)
  print(' discreteState: ', discreteState)
  print(' qTable[discreteState]: ', qTable[discreteState]) # Starting Vals
  print(' np.argmax(qTable[discreteState]): ', np.argmax(qTable[discreteState])) # Starting Vals

  '''
  isDone = False
  while not isDone:
    action = 2
    currState, myReward, isDone, _ = env.step(action)
    print(myReward, currState)
    env.render()

  env.close()
  '''

def getDiscreteState(state, obsSpaceLow, discreteObsWinSize):
  discreteState = (state - obsSpaceLow) / discreteObsWinSize
  return tuple(discreteState.astype(np.int))

if __name__ == "__main__":
  main()