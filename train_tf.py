import json
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
BASE_LR = 1e-3
EPOCHS_FROZEN = 5
EPOCHS_FT = 10

train_dir = "datasets/jawadali1045/20k-multi-class-crop-disease-images/versions/1/Train"
val_dir = "datasets/jawadali1045/20k-multi-class-crop-disease-images/versions/1/Validation"

train_ds = keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)
val_ds = keras.utils.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
])

base = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base.trainable = False  
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = data_augmentation(x)
x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=BASE_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
    keras.callbacks.ModelCheckpoint("models/mobilenetv2_best.keras", save_best_only=True, monitor="val_accuracy"),
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN, callbacks=callbacks)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=BASE_LR * 0.1),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=callbacks)

model.save("models/mobilenetv2_final")
with open("models/preprocessing.json", "w") as f:
    json.dump({
        "img_size": IMG_SIZE,
        "mean": [0, 0, 0], 
        "std": [1, 1, 1],
        "class_names": class_names
    }, f, indent=2)