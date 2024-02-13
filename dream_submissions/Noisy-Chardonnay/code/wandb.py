# Experiment tracking using W&B
from wandb.keras import WandbCallback
import wandb
import tensorflow as tf

architecture_type = ["cnn", "cnn-bilstm", "bilstm-cnn", "cnn-bilstm-attention"]
#wandb.init(project=architecture_type, entity="onuralp", name="model-name")

sweep_config = {
    'method': 'random'
}

metric = {'name': 'loss',
          'goal': 'minimize'}

sweep_config['metric'] = metric

parameters_dict = {

    'dropout_rate': {
        'values': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
    'learning_rate': {
        'distribution': 'uniform',
        'min': 0.001,
        'max': 0.03
    },
    'batch_size': {
        'values': [512, 1024, 2048, 3072, 4096]
    },
    'num_lstm_units': {
        'values': [8, 16, 32, 64, 96, 128]
    },

    'num_attention_heads': {
        'values': [8, 12]
    },

    'num_dense_units': {
        'values': [64, 128, 256, 512, 1024]
    }
}

sweep_config['parameters'] = parameters_dict

# register sweep
sweep_id = wandb.sweep(sweep_config, project='run-name')


def train(config=None):

    with wandb.init(config=config):
        config = wandb.sweep_config
        run_name = wandb.run.name

        learning_rate = config.learning_rate
        dropout_rate = config.dropout_rate
        batch_size = config.batch_size
        num_lstm_units = config.num_lstm_units
        num_attention_heads = config.num_attention_heads
        num_dense_units = config.num_dense_units

        # fixed parameters
        epochs = 10
        sequence_length = 231  # insert + distal and proximal regions
        input_shape = (sequence_length, 4)
        model_path = "./models/{}".format(run_name)

        # fetch model using parameters
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv1D(
            filters=32, kernel_size=3, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(units=num_dense_units)(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

        outputs = tf.keras.layers.Dense(units=1, activation='linear')(x)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

        train_data = tf.data.Dataset.from_tensor_slices((trX, trY_scaled))
        val_data = tf.data.Dataset.from_tensor_slices((valX, valY_scaled))

        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_r_square', verbose=1,
                                                        save_best_only=True, mode='max')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_r_square', patience=10,
                                                          verbose=1, mode='max')

        # add `cvslogger` to save training history
        callbacks_list = [checkpoint, early_stopping, WandbCallback()]

        # set `steps_per_execution` when running on TPU
        model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks_list,
                  batch_size=batch_size, verbose=1, validation_steps=len(val_data),
                  shuffle=True)
