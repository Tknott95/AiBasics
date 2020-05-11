import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

'''
  Switch w/ matplotlib.plypot on linux
  matplotlib w/out jupyter use -> plt.show()
'''

def main():
  print("TF-Verison: ", tf.__version__)
  plt.ion() # start interactive mode. Run: plt.ioff() when finished

  """ 1) IMPORT THE DATA """
  mnist_dataset = keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()
  '''
    The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255.
    The labels are an array of integers, ranging from 0 to 9.
    These correspond to the class of clothing the image represents:
  '''
  class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
  ]

  """ 2) EXPLORE THE DATA """
  print("\ntrain_images.shape: ", train_images.shape)     # (60000, 28, 28)
  print("len(train_labels)): ", len(train_labels))        # 60000
  print("train_labels: ", train_labels)                   # [9 0 0 ... 3 0 5] - each an int between 0-9
  print('test_images.shape: ', test_images.shape)         # (10000, 28, 28)
  print('len(test_labels): ', len(test_labels))           # 10000

  """ 3) PREPROCESS THE DATA """
  plt.figure()
  plt.imshow(train_images[0])
  plt.colorbar()
  plt.grid(False)
  plt.show()    # inspect and see that the pixel values fall in the range of 0 to 255

  '''
    Scale these values to a range of 0 to 1 before feeding them to the neural network model.
    To do so, divide the values by 255.
    It's important that the training set and the testing set be preprocessed in the same way
  '''
  train_images = train_images / 255.0  # ->  /= 255.0 | bug? output array is read-only
  test_images = test_images / 255.0    # ->  /= 255.0 | bug? output array is read-only
  ''' 
    Display the first 25 images from the training set and display the class name below each image. 
  '''
  plt.figure(figsize=(10,10))
  for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
  
  plt.show()

  """ 4) BUILD THE MODEL """


  # Turn off interactive plotting
  plt.ioff()
  plt.show()


if __name__ == '__main__':
  main()
