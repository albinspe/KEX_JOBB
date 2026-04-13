import os
import gc
import numpy as np

from datasets import load_from_disk, concatenate_datasets

import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

BATCH_SIZE = 32
NUM_CORES = 4
IMG_SHAPE = (448, 448, 3)


def create_dataset():
    DATA_DIR = os.path.expanduser("openfake_arrow_chunks_50_50")

    chunk_dirs = sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if d.startswith("chunk_")
    ])

    dataset_list = [load_from_disk(d) for d in chunk_dirs]
    full_dataset = concatenate_datasets(dataset_list)

    return full_dataset


def preprocess_tf_data(img, label):
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    label = tf.cast(label, tf.float32)
    return img, label


def create_tf_pipeline(hf_dataset, is_training=False):
    n = len(hf_dataset)

    def gen():
        read_batch = 64  # intern läs-batch för snabbare Arrow-dekoding
        for start in range(0, n, read_batch):
            end = min(start + read_batch, n)
            batch = hf_dataset[start:end]
            images = batch["image"]
            labels = batch["label"]
            for img, lbl in zip(images, labels):
                yield np.asarray(img, dtype=np.uint8), lbl

    tf_ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )

    if is_training:
        tf_ds = tf_ds.shuffle(
            buffer_size=2000,
            reshuffle_each_iteration=True,
        )

    tf_ds = tf_ds.map(preprocess_tf_data, num_parallel_calls=NUM_CORES)
    tf_ds = tf_ds.batch(BATCH_SIZE)
    tf_ds = tf_ds.prefetch(2)

    return tf_ds


def train_model(train_data, val_data):
    base_model = ResNet50V2(
        weights="imagenet", include_top=False, input_shape=IMG_SHAPE
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks=[early_stopping],
    )

    model.save("Openfake_CNN_model.keras")
    print("Modell sparad")


def main():
    ds = create_dataset()
    ds = ds.class_encode_column("label")

    split_test = ds.train_test_split(
        test_size=0.20, stratify_by_column="label", seed=42
    )
    test_hf = split_test["test"]

    split_val = split_test["train"].train_test_split(
        test_size=0.20, stratify_by_column="label", seed=42
    )
    train_hf = split_val["train"]
    val_hf = split_val["test"]

    test_hf.save_to_disk("openfake_test_split")
    print("Test-set sparat")

    # Skapa TF-pipelines (läser lat från Arrow)
    train_dataset = create_tf_pipeline(train_hf, is_training=True)
    val_dataset = create_tf_pipeline(val_hf)

    # Frigör HF-objekt vi inte längre behöver
    del ds, split_test, split_val, test_hf, train_hf, val_hf
    gc.collect()

    train_model(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
