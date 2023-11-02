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
# TODO やり方古いので、今のやり方にする
class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        "各エポック終了時に重みを保存する"
        model.save("weight.hdf5")
        # tf.keras.saving.save_model(model, "data/04_s2_savedmodel/")


cb = Callback()
initial_epoch = 0
model.fit(
    x=g.generator(),
    validation_data=g.generator_validation_data(),
    validation_steps=s2_data.TEST_BATCH_COUNT,
    # callbacks=[cb],
    steps_per_epoch=s2_data.TRAIN_BATCH_COUNT,
    epochs=15,
    initial_epoch=initial_epoch,
)
