import tensorflow as tf
import h5py
import pandas as pd
from pickle import load
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="path to saved model",
                    default="model/noisy_chardonnay_model.h5", type=Path)
parser.add_argument("--test_seq_onehot", help="path to one hot encoded test data",
                    default="processed/test_seq.h5", type=Path)
parser.add_argument("--test_sequence", help="path to test sequences provided by organizers",
                    default="data/test_sequences.txt", type=Path)
parser.add_argument("--output", help="path to output",
                    default="model/predictions.txt", type=Path)
args = parser.parse_args()

# Reload model from disk
model = tf.keras.models.load_model(args.model_path)

# Check its architecture for visual inspection
print(model.summary())

# Load test data
with h5py.File(args.test_seq_onehot, 'r') as hf:
    teX = hf['test_seq'][:]

# Sanity check on test data
# Shape should be (71103, 110, 4)
print(teX.shape)

# Evaluate the model on the test set
predicted = model.predict(teX[:], batch_size=len(teX))

# Rescale the predicted values using previous StandardScaler
dream_scaler = load(open('model/dream_scaler.pkl', 'rb'))
predicted_inv_scaled = dream_scaler.inverse_transform(predicted)

# Read test sequences into pandas dataframe
predictions_df = pd.read_csv(args.test_sequence, header=None, sep='\t',
                             names=['seq', 'exp'], usecols=['seq'])

# Save the predicted values to a file
predictions_df['pred'] = predicted_inv_scaled[:, 0]
print(predictions_df.head(5))

predictions_df.to_csv(args.output, index=False, sep='\t', header=False)
