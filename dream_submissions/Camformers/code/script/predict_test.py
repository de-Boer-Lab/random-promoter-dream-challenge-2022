"""
Use a saved model to predict a datasets. 

This script will also generate a submission file for the challenge.

Model settings are imported from the main train script.
Make sure you import the version of the main script you want to use and 
that this script is in the same folder.

usage:
python predict_test.py
"""


__date__ = "30/07/2022"


### imports ###
from scipy import stats
#from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

from camformers_submission import * # Make sure you import the version used to train the model.

### configs ###
input_file = "Data/test_sequences.txt" # Datafiles to predict
model_path = "Models/trained_model.pt" # Model to use for prediction
sub_path = "Output/submission_file.csv" # Path to save the predictions
sample_submission = "script/sample_submission.json"


### functions ###

def onehote_reverse(seq):
    """
    Get back sequence from ohe.
    """
    seq2=[] # empty list to store the endoded seq.
    mapping = {"[1.0, 0.0, 0.0, 0.0]":"A", "[0.0, 1.0, 0.0, 0.0]":"C", "[0.0, 0.0, 1.0, 0.0]":"G", "[0.0, 0.0, 0.0, 1.0]":"T"}
    for i in seq:
        i = str(i)
        seq2.append(mapping[i] if i in mapping.keys() else "N") # If not in the above map, use N
    return seq2


### main ###

X, y = OHE(input_file)

X_Tensor=torch.tensor(X)
X_Tensor = X_Tensor.unsqueeze(1)
X_Tensor = torch.transpose(X_Tensor,1,3)
feature_height = X_Tensor.shape[2] # Get the height of the input
feature_width = X_Tensor.shape[3]
Y_Tensor=Tensor(y)

TestLoader = DataLoader(dataset=TensorDataset(X_Tensor, Y_Tensor), batch_size=batch_size, shuffle=False, drop_last=False)

# Load the specified model
model_args = {
        "feature_height":feature_height,
        "feature_width":feature_width,
        "batch_size":batch_size,
        "print_size":True,
        "out_channels":out_channels,
        "kernels":kernels,
        "pool_kernels":pool_kernels,
        "paddings":paddings,
        "strides":strides,
        "pool_strides":pool_strides,
        "dropouts":dropouts,
        "linear_output":linear_output,
        "linear_dropouts":linear_dropouts
    }
Model_best = CNN(**model_args)
Model_best.load_state_dict(torch.load(model_path))
Model_best.to(device)

# Predict the loaded dataset.
y_pred, y_true = Model_Pred(Model_best, TestLoader)

# Save sequence and prediction results to a file.
subfile = open(sub_path, "w")
for n, pred in enumerate(y_pred):
    seq = "".join(onehote_reverse(X[n]))
    new_line = "%s\t%s\n" % (seq, pred)
    subfile.write(new_line)
subfile.close()

df_describe = pd.DataFrame(y_pred)
print(df_describe.describe())
print("< 0:\t",len([x for x in y_pred if x < 0]))

# Create the submission file
with open(sample_submission, 'r') as f:
    ground = json.load(f)

indices = np.array([int(indice) for indice in list(ground.keys())])
PRED_DATA = OrderedDict()

for i in indices:
#Y_pred is an numpy array of dimension (71103,) that contains your
#predictions on the test sequences
    PRED_DATA[str(i)] = float(y_pred[i])

def dump_predictions(prediction_dict, prediction_file):
    with open(prediction_file, 'w') as f:
        json.dump(prediction_dict, f)

timestr = time.strftime('%Y%m%d-%H%M%S')
dump_predictions(PRED_DATA, 'submission'+timestr+'.json')

print('Submission file "'+'submission'+timestr+'.json'+'" has been prepared.')
