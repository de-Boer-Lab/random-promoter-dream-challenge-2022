import os, sys, yaml, importlib, datetime, argparse
from utils.train_utils import *

for _path in path_to_pad:
    sys.path.append(_path)

import numpy as np
import pandas as pd
from utils.utils import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--config", type=str, dest="config", action="store", help="Config name to run"
)
parser.add_argument(
    "--model",
    type=str,
    dest="model",
    action="store",
    help="model type, override config file model type",
    default="",
)

parser.add_argument(
    "--reset",
    dest="reset",
    action="store_true",
    help="reset train,valid,test indeces",
    default=False,
)
parser.add_argument(
    "--train_only",
    dest="train_only",
    action="store_true",
    help="only train model",
    default=False,
)
parser.add_argument(
    "--predict_only",
    dest="predict_only",
    action="store_true",
    help="only make prediction",
    default=False,
)

args = parser.parse_args()

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)


def run_model(config_name):

    # handle params
    params_file = "configs/{}.yaml".format(
        config_name
    )
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    model_params = params["model"]
    pre_train_params = params["pre_train"]
    fine_tune_params = params["fine_tune"]
    preprocessing_params = params["preprocessing"]

    # load model specific module
    if args.model != "":
        if params["model_type"] != args.model:
            params["model_type"] = args.model

    model_module = importlib.import_module(params["model_type"])

    # initialize model
    with strategy.scope():
        model = model_module.build(**model_params)
        if pre_train_params["learning_rate_schedule"] == True:
            lr_schedule = getattr(
                tf.keras.optimizers.schedules, pre_train_params["schedule_name"]
            )(**pre_train_params["schedule"])
        else:
            lr_schedule = pre_train_params["learning_rate"]

        optimizer = getattr(tf.keras.optimizers, pre_train_params["optimizer_type"])(
            lr_schedule
        )
        if "loss_name" in pre_train_params.keys():
            loss = getattr(tf.keras.losses,pre_train_params["loss_name"])(
                **pre_train_params["loss"]
            )
            _pre_train_loss=pre_train_params["loss_name"]
            print("using {} loss".format(pre_train_params["loss_name"]))
        else:
            loss = tf.keras.losses.MSE
            _pre_train_loss="MSE"
            print("using {} loss".format("MSE"))
        #
        model.compile(
            optimizer=optimizer,
            loss=loss,
            steps_per_execution=64,
            metrics=[RSquare_CoD1, Pearson_r, Weighted_metrics_tpu],
        )

    # callbacks
    es_callback_pre_train = tf.keras.callbacks.EarlyStopping(
        monitor="val_Weighted_metrics_tpu", patience=3, restore_best_weights=True, mode="max"
    )
    mc_callback_pre_train = tf.keras.callbacks.ModelCheckpoint(
        filepath="model/{}/{}/best_weights.h5".format(
            params["model_type"], config_name
        ),
        monitor="val_Weighted_metrics_tpu",
        save_best_only=True,
        save_weights_only=True,
        save_freq="epoch",
        mode="max",
    )
    tb_callback_pre_train = tf.keras.callbacks.TensorBoard(
        log_dir="tensorboard/{}/{}/pre_train".format(params["model_type"], config_name)
    )

    # dirs
    os.makedirs("model", exist_ok=True)
    os.makedirs("model/{}".format(params["model_type"]), exist_ok=True)
    os.makedirs("model/{}/{}".format(params["model_type"], config_name), exist_ok=True)
    os.makedirs("tensorboard", exist_ok=True)
    os.makedirs("tensorboard/{}".format(params["model_type"]), exist_ok=True)
    os.makedirs(
        "tensorboard/{}/{}".format(params["model_type"], config_name), exist_ok=True
    )
    os.makedirs(
        "tensorboard/{}/{}/pre_train".format(params["model_type"], config_name),
        exist_ok=True,
    )
    os.makedirs(
        "tensorboard/{}/{}/fine_tune".format(params["model_type"], config_name),
        exist_ok=True,
    )
    os.makedirs("prediction", exist_ok=True)
    os.makedirs("prediction/{}".format(params["model_type"]), exist_ok=True)
    os.makedirs(
        "prediction/{}/{}".format(params["model_type"], config_name), exist_ok=True
    )
    os.makedirs("train_valid_data",exist_ok=True)
    # load df for splitting
    df = pd.read_csv(
        os.path.join(data_path, "train_sequences.txt"),
        header=None,
        names=["seq", "y"],
        sep="\t",
    )
    df["y"] = df["y"].astype(np.float32)
    if "sample_weight_limit" in params.keys():
        df, class_weight_dict = add_sample_weights(df, limit=params["sample_weight_limit"])
    else:
        df, class_weight_dict = add_sample_weights(df, limit=3)

    y_mean = df["y"].mean()
    y_sd = df["y"].std()

    df["y"] = [(x - y_mean) / y_sd for x in df["y"]]

    # load index
    if args.reset == False:
        if (
                (os.path.isfile("train_valid_data/train_idx.npy"))
                and (os.path.isfile("train_valid_data/valid_idx.npy"))
                and (os.path.isfile("train_valid_data/test_idx.npy"))
        ):
            train_idx = np.load("train_valid_data/train_idx.npy")
            valid_idx = np.load("train_valid_data/valid_idx.npy")
            test_idx = np.load("train_valid_data/test_idx.npy")
        else:
            train_idx, valid_idx = train_test_split(df.index.tolist(),
                                                    test_size=0.35,
                                                    stratify=df["cls"]
                                                    )
            valid_idx, test_idx = train_test_split(valid_idx,
                                                   test_size=0.7,
                                                   stratify=df.iloc[valid_idx]["cls"]
                                                   )
            np.save("train_valid_data/train_idx.npy", train_idx)
            np.save("train_valid_data/valid_idx.npy", valid_idx)
            np.save("train_valid_data/test_idx.npy", test_idx)
    else:
        train_idx, valid_idx = train_test_split(df.index.tolist(),
                                                test_size=0.35,
                                                stratify=df["cls"]
                                                )
        valid_idx, test_idx = train_test_split(valid_idx,
                                               test_size=0.7,
                                               stratify=df.iloc[valid_idx]["cls"]
                                               )
        np.save("train_valid_data/train_idx.npy", train_idx)
        np.save("train_valid_data/valid_idx.npy", valid_idx)
        np.save("train_valid_data/test_idx.npy", test_idx)

    # generators

    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    train_data = tf.data.Dataset.from_generator(
        TrainDataGen_with_sample_weights(
            df=df.iloc[train_idx],
            batch_size=params["batch_size"],
            **preprocessing_params
        ),
        output_signature=(
            tf.TensorSpec(
                shape=(preprocessing_params["fix_length"], 4), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    ).with_options(options)
    valid_data = tf.data.Dataset.from_generator(
        TrainDataGen_with_sample_weights(
            df=df.iloc[valid_idx],
            batch_size=params["batch_size"],
            **preprocessing_params
        ),
        output_signature=(
            tf.TensorSpec(
                shape=(preprocessing_params["fix_length"], 4), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    ).with_options(options)
    test_data = tf.data.Dataset.from_generator(
        TrainDataGen_with_sample_weights(
            df=df.iloc[test_idx],
            batch_size=params["batch_size"],
            **preprocessing_params
        ),
        output_signature=(
            tf.TensorSpec(
                shape=(preprocessing_params["fix_length"], 4),
                dtype=tf.float32
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    ).with_options(options)



    # Kmer injection (use 5-mers and 6-mers)
    _5mer_list = gen_kmer_list(5)
    _6mer_list = gen_kmer_list(6)
    _5mer_motif = motif.load_kmers(_5mer_list)
    _6mer_motif = motif.load_kmers(_6mer_list)
    kmer_motif = np.concatenate([_5mer_motif, _6mer_motif])
    kmer_motif_kernel = motif.motif_to_kernel(
        motif_vec=kmer_motif,
        include_rc=False,
        predefined_length=model_params["kmer_length_max"],
    )
    insert_kernels(
        model=model, target_layer_name="Scanning/kmer", kernel=kmer_motif_kernel
    )

    if args.predict_only==False:

        # model fitting

        print("Pre-train")
        # fit current data
        model.fit(
            train_data.cache()
            .shuffle(params["batch_size"] * 10, reshuffle_each_iteration=True)
            .repeat(pre_train_params["max_epochs"])
            .batch(params["batch_size"], drop_remainder=True)
            .prefetch(64), # tpu version, as it runs on 8*8 blocks, so it should be 64 or 128
            validation_data=valid_data.cache() # never changes during pseudo-labeling phase
            .repeat(pre_train_params["max_epochs"])
            .batch(params["batch_size"], drop_remainder=False)
            .prefetch(64),
            epochs=pre_train_params["max_epochs"],
            workers=15,
            max_queue_size=45,
            use_multiprocessing=True,
            verbose=1,
            steps_per_epoch= len(train_idx) // params["batch_size"],
            validation_steps=np.ceil(len(valid_idx) / params["batch_size"]),
            callbacks=[
                es_callback_pre_train,
                mc_callback_pre_train,
                tb_callback_pre_train
            ],
        )

        # fine-tune
        # re-initialize model
        with strategy.scope():
            model = model_module.build(**model_params)
            if fine_tune_params["learning_rate_schedule"] == True:
                lr_schedule = getattr(
                    tf.keras.optimizers.schedules, fine_tune_params["schedule_name"]
                )(**fine_tune_params["schedule"])
            else:
                lr_schedule = fine_tune_params["learning_rate"]

            optimizer = getattr(tf.keras.optimizers, fine_tune_params["optimizer_type"])(
                lr_schedule
            )
            if "loss_name" in fine_tune_params.keys():
                loss = getattr(tf.keras.losses, fine_tune_params["loss_name"])(
                    **fine_tune_params["loss"]
                )
                _fine_tune_loss_name = fine_tune_params["loss_name"]
                print("using {} loss".format(fine_tune_params["loss_name"]))
            else:
                loss = tf.keras.losses.MSE
                print("using {} loss".format("MSE"))
                _fine_tune_loss_name = "MSE"
            #
            model.compile(
                optimizer=optimizer,
                loss=loss,
                steps_per_execution=64,
                metrics=[RSquare_CoD1, Pearson_r, Weighted_metrics_tpu],
            )

        # load pre-train weights
        model.load_weights("model/{}/{}/best_weights.h5".format(
            params["model_type"], config_name
            ),
        )

        # relax kmer weights
        for idx, layer in enumerate(model.layers):
            if layer.name == "Scanning/kmer":
                model.layers[idx].trainable = True
                break

        # callbacks
        es_callback_fine_tune = tf.keras.callbacks.EarlyStopping(
            monitor="val_Weighted_metrics_tpu", patience=10, restore_best_weights=True, mode="max"
        )
        mc_callback_fine_tune = tf.keras.callbacks.ModelCheckpoint(
            filepath="model/{}/{}/fine_tune_v7_best_weights.h5".format(
                params["model_type"], config_name
            ),
            monitor="val_Weighted_metrics_tpu",
            save_best_only=True,
            save_weights_only=True,
            save_freq="epoch",
            mode="max",
        )
        tb_callback_fine_tune = tf.keras.callbacks.TensorBoard(
            log_dir="tensorboard/{}/{}/fine_tune_v7".format(params["model_type"], config_name)
        )

        # load initial indeces
        train_idx = np.load("train_valid_data/train_idx.npy")
        valid_idx = np.load("train_valid_data/valid_idx.npy")
        test_idx = np.load("train_valid_data/test_idx.npy")
        fine_tune_test_idx, fine_tune_valid_idx=train_test_split(train_idx,
                                                                 test_size=0.25,
                                                                 stratify=df.iloc[train_idx]["cls"]
                                                                 )
        fine_tune_train_idx=np.concatenate([valid_idx,test_idx,fine_tune_test_idx])

        # generators
        train_data = tf.data.Dataset.from_generator(
            TrainDataGen_with_sample_weights(
                df=df.iloc[fine_tune_train_idx],
                batch_size=params["batch_size"],
                **preprocessing_params
            ),
            output_signature=(
                tf.TensorSpec(
                    shape=(preprocessing_params["fix_length"], 4), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        ).with_options(options)
        valid_data = tf.data.Dataset.from_generator(
            TrainDataGen_with_sample_weights(
                df=df.iloc[fine_tune_valid_idx],
                batch_size=params["batch_size"],
                **preprocessing_params
            ),
            output_signature=(
                tf.TensorSpec(
                    shape=(preprocessing_params["fix_length"], 4), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        ).with_options(options)


        # learn
        model.fit(
            train_data.cache()
            .shuffle(params["batch_size"] * 10, reshuffle_each_iteration=True)
            .repeat(fine_tune_params["epochs"]*10)
            .batch(params["batch_size"], drop_remainder=True)
            .prefetch(64), # tpu version, as it runs on 8*8 blocks, so it should be 64 or 128
            validation_data=valid_data.cache() # never changes during pseudo-labeling phase
            .repeat(fine_tune_params["epochs"]*10)
            .batch(params["batch_size"], drop_remainder=False)
            .prefetch(64),
            epochs=fine_tune_params["epochs"]*10,
            workers=15,
            max_queue_size=45,
            use_multiprocessing=True,
            verbose=1,
            steps_per_epoch= len(train_idx) // params["batch_size"],
            validation_steps=np.ceil(len(fine_tune_valid_idx) / params["batch_size"]),
            callbacks=[
                es_callback_fine_tune,
                mc_callback_fine_tune,
                tb_callback_fine_tune
            ],
        )

    if args.train_only!=True:

        # make prediction
        # load best fine tune weights
        model.load_weights("model/{}/{}/fine_tune_v7_best_weights.h5".format(
            params["model_type"], config_name
            ),
        )
        # predict df
        predict_df = pd.read_csv(
            os.path.join(data_path, "test_sequences.txt"), # todo: change to final submission file
            header=None,
            names=["seq", "y"],
            sep="\t",
        )
        #
        predict_data = tf.data.Dataset.from_generator(
            PredictDataGen(
                df=predict_df,
                batch_size=params["batch_size"],
                **preprocessing_params
            ),
            output_signature=(
                tf.TensorSpec(
                    shape=(preprocessing_params["fix_length"], 4),
                    dtype=tf.float32
                )
            ),
        ).with_options(options)

        # prediction
        predict_df["y"] = model.predict(
            predict_data.batch(params["batch_size"], drop_remainder=False),
            workers=15,
            max_queue_size=45,
            use_multiprocessing=True,
            verbose=1,
            steps=np.ceil(predict_df.shape[0] / params["batch_size"]),
        )

        # re-scaling & output
        predict_df["y"]=[x*y_sd+y_mean for x in predict_df["y"]]
        predict_df.to_csv(
            "prediction/{}/{}/test_sequences_fine_tune_v7_{}_{}.txt"
            .format(
                params["model_type"],
                config_name,
                config_name,
                datetime.datetime.today().strftime(format="%Y_%m_%d_%H_%M_%S"),
            ),
            header=False,
            index=False,
            sep="\t",
        )


    tf.tpu.experimental.shutdown_tpu_system(cluster_resolver)


run_model(args.config)
