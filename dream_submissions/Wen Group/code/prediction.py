import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow_addons as tfa

r_square = tfa.metrics.r_square.RSquare(dtype=tf.float32, y_shape=(1,))

if tf.config.list_physical_devices('GPU'):
    strategy = tf.distribute.MirroredStrategy()
    print("All devices: ", tf.config.list_logical_devices('GPU'))
    # # for gpu
    with strategy.scope():
        model = tf.keras.models.load_model('saved_model/tpu_trans_unet_v0.54', custom_objects={'r_square':r_square})
else:  
    # Use the TPU Strategy
    ##TPU
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(resolver)

    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    
    strategy = tf.distribute.TPUStrategy(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    # # for tpu
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    with strategy.scope():
        model = tf.keras.models.load_model('saved_model/tpu_trans_unet_v0.54', custom_objects={'r_square':r_square},options=load_options)

model.summary()



# read data
data_path="data/test_sequences.txt"
df = pd.read_csv(data_path, sep="\t", header=None, names=["Seq", "Expression"])

dataset = tf.data.TextLineDataset(data_path)

seq_list=[]
exp_list=[]
pbar = tqdm(total=len(df))
    
for i in dataset.as_numpy_iterator():
  seq, exp = i.decode("utf-8").split("\t")
  seq_list.append(list(seq))
  exp_list.append(float(exp))
  pbar.update(1)

pbar.close()

# "padding" the test data so as to make sure each batch contains 50 examples
seq_list = seq_list + seq_list[:47]
exp_list = exp_list + exp_list[:47]
print(len(seq_list))

# pad sequence to 112 for trans_unet
pad_seq_list = tf.keras.preprocessing.sequence.pad_sequences(seq_list, maxlen=112, padding="post", truncating='post', dtype="str", value="N")

new_dataset = tf.data.Dataset.from_tensor_slices((pad_seq_list,exp_list))

vocab = ['A','C','G','T']
lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
new_dataset = new_dataset.map(lambda x, y: (lookup(x), y))

new_dataset = new_dataset.map(lambda x, y: (tf.cast(x, dtype=tf.float32), y))
new_dataset = new_dataset.batch(50, drop_remainder=True)

pred = model.predict(new_dataset)

# Discard the "padding" samples
df["Expression"] = pred[:71103]

# previous submission requirement
df.to_csv('submission/submission.txt', sep="\t", header=False, index=False)