import tensorflow as tf
import s2_data
from s2_data import S2Data

g = S2Data()
model: tf.keras.Model = tf.keras.applications.EfficientNetV2B3(
    input_shape=(224, 224, 3),
    weights=None,  # type: ignore
    classes=5,
    classifier_activation="softmax",
)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


# コールバック
class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        "各エポック終了時に重みを保存する"
        model.save("weight.hdf5")


cb = Callback()
initial_epoch = 0
model.fit_generator(
    g.generator(),
    validation_data=g.generator_test(),
    validation_steps=g.test_steps(),
    callbacks=[cb],
    steps_per_epoch=s2_data.TRAIN_SIZE / s2_data.BATCH_SIZE,
    epochs=20,
    initial_epoch=initial_epoch,
)
