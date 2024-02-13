import click
import numpy as np
import pandas as pd

import random
import os

from sequence import SeqDreamChallange1D

from model import (
    standard,
    simplified,
    ResNet1D_Classification,
    ResNet1D_Regression,
    ResidualUnit1D_BN,
    ResNet1D_Classification_medium,
    ResNet1D_Regression_medium,
    ResNet1D_Classification_dropout,
    ResNet1D_Regression_dropout,
)

import tensorflow as tf
from tensorflow.keras import Model, activations, layers
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    GlobalAveragePooling1D,
    add,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers, callbacks

SEED = 0

def set_seeds(seed=SEED):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    
    

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)





model_type = {
    "standard": standard,
    "simplified": simplified,
    "ResNet1D_Classification": ResNet1D_Classification,
    "ResNet1D_Regression": ResNet1D_Regression,
    "ResNet1D_Classification_medium": ResNet1D_Classification_medium,
    "ResNet1D_Regression_medium": ResNet1D_Regression_medium,
    "ResNet1D_Classification_dropout": ResNet1D_Classification_dropout,
    "ResNet1D_Regression_dropout": ResNet1D_Regression_dropout,
}

# options
@click.command()
@click.option(
    "--train",
    "train_tsv_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Training sequences",
)
@click.option(
    "--val",
    "val_tsv_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Validation sequences",
)
@click.option(
    "--model",
    "model_file",
    required=True,
    type=click.Path(writable=True),
    help="Model output file",
)
@click.option(
    "--weights",
    "weights_file",
    required=True,
    type=click.Path(writable=True),
    help="Weights output file",
)
@click.option(
    "--val-acc",
    "acc_file",
    required=True,
    type=click.Path(writable=True),
    help="Accuracy validation output file",
)
@click.option(
    "--val-pred",
    "pred_file",
    required=False,
    type=click.Path(writable=True),
    help="Prediction validation output file",
)
@click.option(
    "--fit-log",
    "fit_log_file",
    required=True,
    type=click.Path(writable=True),
    help="Fit log file",
)
@click.option(
    "--batch-size",
    "batch_size",
    required=False,
    default=32,
    type=int,
    help="Batch size",
)
@click.option(
    "--label-columns",
    "label_columns",
    required=False,
    default=(2,None),
    type=(int,int),
    help="label start and stop columns ",
)
@click.option(
    "--model-type",
    "model_type_str",
    default="standard",
    type=click.Choice(model_type.keys(), case_sensitive=False),
    help="The model that should be used.",
)
@click.option(
    "--tensorboard-folder",
    "tensorboard_dir",
    required=False,
    type=click.Path(),
    help="Tensorboard folder",
)
@click.option(
    "--SavedModel",
    "savedmodel_dir",
    required=False,
    type=click.Path(),
    help="Folder for saving model ins tensorflow SavedModel format",
)
@click.option("--epochs", "epochs", required=True, type=int, help="Number of epochs")
@click.option(
    "--learning-rate", "learning_rate", required=True, type=float, help="Learning rate"
)
@click.option(
    "--use-learning-rate-sheduler/--no-learning-rate-sheduler",
    "learning_rate_sheduler",
    default=False,
    help="Learning rate sheduler",
)
@click.option(
    "--use-early-stopping/--no-early-stopping",
    "early_stopping",
    default=False,
    help="Learning rate",
)
@click.option(
    "--loss",
    "loss",
    type=click.Choice(
        [
            "MSE",
            "Huber",
            "Poission",
            "CategoricalCrossentropy",
        ],
        case_sensitive=False,
    ),
    default="MSE",
    help="Choise of loss function.",
)
@click.option(
    "--model-mode",
    "model_mode",
    type=click.Choice(
        [
            "classification",
            "regression",
        ],
        case_sensitive=False,
    ),
    default='classification',
    required=False,
    help="Choise of model type",
)
@click.option(
    "--adapter-trimming/--no-adapter-trimming",
    "adapter_trimming",
    default=False,
    help="Trim adapters from the sequence",
)
@click.option('--seed',
              'seed',
              required=False,
              type=int,
              default=None,
              help='seed for randomness.'
)
@click.option(
    "--fit_sequence",
    "fit_seq",
    is_flag=True,
    default=None,
    help="fit sequences to sequence length (110bp)"
)
# use_poisson_loss,use_huber_loss,use_cat_ent
def cli(
    train_tsv_file,
    val_tsv_file,
    batch_size,
    model_file,
    weights_file,
    pred_file,
    acc_file,
    fit_log_file,
    tensorboard_dir,
    savedmodel_dir,
    epochs,
    learning_rate,
    learning_rate_sheduler,
    early_stopping,
    loss,
    label_columns,
    model_mode,
    adapter_trimming,
    model_type_str,
    seed,
    fit_seq
):
    if seed:
        set_global_determinism(seed)

    if tensorboard_dir:
        try:
            # Create target Directory
            os.makedirs(tensorboard_dir, exist_ok = True)
            print("Directory ", tensorboard_dir, " Created ")
        except FileExistsError:
            print("Directory ", tensorboard_dir, " already exists")
        except OSError as error:
            print(f"Directory {tensorboard_dir} can not be created")
            print(error)

    strategy = tf.distribute.MirroredStrategy(devices=None)

    def lr_schedule(epoch):
        """
        Returns a custom learning rate that decreases as epochs progress.
        """
        if epoch > 10:
            learning_rate = 0.00001

        return learning_rate
    start_column, stop_column = label_columns
    # load DATA
    if model_mode == 'regression':
        dl = SeqDreamChallange1D(
            train_tsv_file,
            label_start_column=start_column,
            label_stop_column=stop_column,
            label_dtype=float,
            trim_adapters=adapter_trimming,
            
        )
        dl_val = SeqDreamChallange1D(
            val_tsv_file,
            label_start_column=start_column,
            label_stop_column=stop_column,
            label_dtype=float,
            trim_adapters=adapter_trimming,
            fit_sequence=fit_seq
        )
    elif model_mode == 'classification':
        dl = SeqDreamChallange1D(
            train_tsv_file,
            label_start_column=start_column,
            label_stop_column=stop_column,
            label_dtype=int,
            trim_adapters=adapter_trimming,
            fit_sequence=fit_seq
        )
        dl_val = SeqDreamChallange1D(
            val_tsv_file,
            label_start_column=start_column,
            label_stop_column=stop_column,
            label_dtype=int,
            trim_adapters=adapter_trimming,
            fit_sequence=fit_seq
        )

    # iterator = dl.batch_train_iter(batch_size=batch_size, num_workers=4)
    val_data = dl_val.load_all()
    train_data = dl.load_all()

    with strategy.scope():

        model = model_type[model_type_str](
            val_data["inputs"].shape, val_data["targets"].shape
        )

        # defining callbacks
        call_backs = []
        if tensorboard_dir:
            tensorboard = callbacks.TensorBoard(
                log_dir=tensorboard_dir,
                histogram_freq=1,
                batch_size=batch_size,
                write_graph=True,
                write_grads=True,
                write_images=True,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None,
                embeddings_data=None,
                update_freq="epoch",
            )
            call_backs.append(tensorboard)

        csvLogger = callbacks.CSVLogger(fit_log_file, separator="\t", append=False)

        call_backs.append(csvLogger)

        lr_callback = callbacks.LearningRateScheduler(lr_schedule)
        if learning_rate_sheduler:
            call_backs.append(lr_callback)

        earlyStopping_callback = callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
        if early_stopping:
            call_backs.append(earlyStopping_callback)

        optimizer = optimizers.Adam(learning_rate=learning_rate)

        if loss == "Poission":  # use_poisson_loss:
            model.compile(
                loss=tf.keras.losses.Poisson(),
                metrics=["mse", "mae", "mape", "acc"],
                optimizer=optimizer,
            )
        elif loss == "Huber":  # use_huber_loss:
            model.compile(
                loss=tf.keras.losses.huber(),
                metrics=["mse", "mae", "mape", "acc"],
                optimizer=optimizer,
            )
        elif loss == "CategoricalCrossentropy":  # use_cat_ent:
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["mse", "mae", "mape", "acc", "categorical_accuracy"],
                optimizer=optimizer,
            )
        else:
            model.compile(
                loss="mean_squared_error",
                metrics=["mse", "mae", "mape", "acc"],
                optimizer=optimizer,
            )

        print("Fit model")
        # result = model.fit(iterator, steps_per_epoch=len(dl)//batch_size,
        #                     validation_data=(val_data["inputs"], val_data["targets"]),
        #                     batch_size=batch_size,
        #                     epochs=epochs,
        #                     shuffle=True,
        #                     verbose=2,
        #                     callbacks=call_backs)

        result = model.fit(
            train_data["inputs"],
            train_data["targets"],
            validation_data=(val_data["inputs"], val_data["targets"]),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            verbose=2,
            callbacks=call_backs,
        )

        print("Save_model")

        if savedmodel_dir:
            model.save(savedmodel_dir)
            print(f"Model saved to: {savedmodel_dir}")



        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(weights_file)

        if pred_file:
            print("Final prediction")
            preds = model.predict(val_data["inputs"])
            pd.DataFrame(preds).to_csv(pred_file, sep="\t", index=False)

        print("Final evaluation")
        eval = model.evaluate(val_data["inputs"], val_data["targets"])
        pd.DataFrame(eval).to_csv(acc_file, sep="\t", index=False, header=None)


if __name__ == "__main__":
    cli()
