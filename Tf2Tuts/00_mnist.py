import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

'''
  Avid notetaking for an inline view of this basic mnist set.
  Every step explains proc in my own words.
  Followed tensorflow.org docs, a François Chollet, the creator of keras, example.
'''

def main():
  print("TF-Verison: ", tf.__version__)
  # plt.ion() # start interactive mode. Run: plt.ioff() when finished

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
  print('len(test_labels): ', len(test_labels), "\n\n")           # 10000

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
  ''' 4-01 Setup the layers '''
  # keras.layers.Flatten(input_shape=(28,28)),
  '''
      This first layer transforms the format of the imgs from a 2D array, of 28 * 28 = 784 pixels.
      "Sort of" unstacks rows of pixels in the img and lines them up.
      Layer has no params to learn, only reformats the data.
      Now that the pixels are flattened the network consists of two tf.layers.Dense layers.
      These will be densely connected, or fully connected, neural layers.
  '''
  # keras.layers.Dense(128, activation='relu'),
  # keras.layers.Dense(10)
  '''
      The first Dense layer has 128 nodes, or neurons. The second (and last) layer returns a logits array with length of 10.
      Each node contains a "score" that indicates the curr img belongs to one of the 10 classes.
  '''
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
  ])
  

  ''' 4-02 Compile the model '''

  '''
    To be added whilst model is training:
      LossFunction — Measures model accuracy during training. 
                      Goal is to minimize this function to guide model in right direction.
      Optimizer    - How the model is updated via. data it sees from loss func
      Metrics      - Monitors training & testing steps. Will use accuracy, the fraction of images correctly classified.
  '''
  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  """ 5) TRAIN THE MODEL """
  '''
    Requires following steps:
      1) Feed training data, train_(images/labels) arrays, to model. 
      2) Model learns image -> label associations
      3) Ask model to make predictions about the test data, test_images array in this scenario, could use my own if wanted
      4) Verify that predictions are correct, they match the labels, from the test_labels array in this scenario.
  '''
  ''' 5-01 Feed da model bby '''
  model.fit(train_images, train_labels, epochs=14)

  ''' 5-02 Test the accuracy, observe if overfitting takes place '''
  test_loss, test_accuracy = model.evaluate(test_images,  test_labels, verbose=2)
  print('\nTest accuracy: ', test_accuracy)
  '''
    After 14 epochs I see my test accuracy, 0.892799973487854, is less than my training accuracy, 0.9224
    This shows I am overfitting.
    An overfitted model memorizes the noise and details of a training dataset to a point where it negatively impacts the performance of the model.
  '''

  ''' 5-03 Go full on nostradamus w/ predictions '''
  # The model's linear outputs, logits, attach a softmax layer for converting into probabilities for easier interpretation.
  prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  predicts = prob_model.predict(test_images)
  print('\npredicts[0]: ', predicts[0])
  '''
    A prediction is an array of 10 numbers of probability to each of the 10 different options.
    To see the top predict run np.argmax(predicts[0])

    0 	T-shirt/top
    1 	Trouser
    2 	Pullover
    3 	Dress
    4 	Coat
    5 	Sandal
    6 	Shirt
    7 	Sneaker
    8 	Bag
    9 	Ankle boot
  '''
  print('\nOption Predicted: ', np.argmax(predicts[0]))

  # Turn off interactive plotting w/ show so plt doesn't auto close
  # plt.ioff()
  # plt.show()


if __name__ == '__main__':
  main()
