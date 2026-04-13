import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input 
from tensorflow.keras import layers, models
import os
from datasets import load_from_disk, concatenate_datasets

DATA_DIR = "openfake_arrow_chunks_50_50"
IMG_SIZE = (448, 448)
BATCH_SIZE = 32
TEST_LIMIT = 1000


chunk_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if d.startswith("chunk_")]
dataset_list = [load_from_disk(d) for d in chunk_dirs]
full_dataset = concatenate_datasets(dataset_list)

subset_dataset = full_dataset.shuffle(seed = 42).select(range(TEST_LIMIT))


def gen():
    for ex in subset_dataset:
        try:
            img = ex["image"].convert("RGB").resize(IMG_SIZE)
            img_array = tf.keras.utils.img_to_array(img)
            img_preprocessed = preprocess_input(img_array)
            label = 1 if ex["label"] == "real" else 0
            yield img_preprocessed, label
        except Exception as e:
            continue 

full_ds = tf.data.Dataset.from_generator(
    gen, 
    output_signature= (
        tf.TensorSpec(shape = (448, 448, 3), dtype = tf.float32),
        tf.TensorSpec(shape = (), dtype = tf.int32)
    )
).cache()

train_ds = full_ds.take(800).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = full_ds.skip(800).take(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

base_model = ResNet50V2(weights = "imagenet", include_top = False, input_shape = (448, 448, 3))
base_model.trainable = False

model = models.Sequential([
    base_model, 
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation = "relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation = "sigmoid")
])

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 3
)

