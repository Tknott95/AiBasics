import functools
import numpy as np
import tensorflow as tf

'''
  @TITLE
    Would I survive the titanic?
  @DOING
   Will be loading pre-made csv data, from daZ g00gle, and playing w/ it.
  @CREDS
    Will have my own tweaks so think of this as a "fork" in a way.
    From tensorflow.ord docs,original author of base code tfTeam.
''' 

def main():
  # Will use my own lcoal files for nexxt csv example. Following tf2 coding conventions prior
  # Have done this type of coding multiple times yet not with tf2 and new conventions
  TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
  TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

  train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
  test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

  # This makes np vals easier to read, @TODO refer to docs and understand why as I have never done this
  np.set_printoptions(precision=2, suppress=True)

  # Viewing how CSV is formatted w/ !head {train_file_path} from docs must be a jupyter func
  # ONLY A JUPYTER FUNC: !head {train_file_path} - Will just create my own to reuse
  ''' SHOWS CSV HEAD AS
    survived,sex,age,n_siblings_spouses,parch,fare,class,deck,embark_town,alone
    0,male,22.0,1,0,7.25,Third,unknown,Southampton,n
    1,female,38.0,1,0,71.2833,First,C,Cherbourg,n
    1,female,26.0,0,0,7.925,Third,unknown,Southampton,y
    1,female,35.0,1,0,53.1,First,C,Southampton,n
    0,male,28.0,0,0,8.4583,Third,unknown,Queenstown,y
    0,male,2.0,3,1,21.075,Third,unknown,Southampton,n
    1,female,27.0,0,2,11.1333,Third,unknown,Southampton,n
    1,female,14.0,1,0,30.0708,Second,unknown,Cherbourg,n
    1,female,4.0,1,1,16.7,Third,G,Southampton,n
  '''
  # Only col I will need w/ label is the one I am predicting, which is: WillISurvive?
  # Following tf2 conventions, aka CAPS
  LABEL_COL = 'survived'
  LABELS = [0,1]

  # Using Custom Function: get_dataset(path, **kwargs)
  raw_train_data = get_dataset(LABEL_COL, train_file_path)
  raw_test_data = get_dataset(LABEL_COL, test_file_path)

  show_batch(raw_train_data)
  
  # Set col labels after seeing data each batches label
  '''
    This is only if data has no col names.
    If this is the case then pass in, column_names=MY_COL_LABELS_FROM_CSV
      MY_COL_LABELS_FROM_CSV = ['survived', 'sex', 'age', ...]
      get_dataset(_path, column_names=MY_COL_LABELS_FROM_CSV)
  '''
  temp_dataset = get_dataset(LABEL_COL, train_file_path)
  show_batch(temp_dataset)


'''
  Helper Functions 
  (not modular in my Tf2KerasRef/*'s)
  These are MVP's
 '''
# Each item in the dataset is a batch, represented as a tuple
def show_batch(dataset):
  print('\n')
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

# For reading csv data and creating dataset
def get_dataset(col_label, _path, **kwargs):
  new_dataset = tf.data.experimental.make_csv_dataset(
    _path,
    batch_size=5,
    label_name=col_label,
    na_value="?",
    num_epochs=1,
    ignore_errors=True,
    **kwargs
  )
  return new_dataset


if __name__ == '__main__':
  main()
