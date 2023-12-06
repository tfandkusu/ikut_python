import pickle
import tensorflow as tf
import s2_data
from s2_data import S2Data


g = S2Data()
# EfficientNetV2B2 を使う
model: tf.keras.Model = tf.keras.applications.EfficientNetV2B2(
    input_shape=(
        s2_data.INPUT_IMAGE_SIZE,
        s2_data.INPUT_IMAGE_SIZE,
        s2_data.INPUT_IMAGE_CHANNEL,
    ),
    weights=None,  # type: ignore
    classes=5,
    classifier_activation="softmax",
)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        "各エポック終了時にモデルを保存する"
        tf.keras.saving.save_model(model, s2_data.MODEL_DIR)


cb = Callback()
initial_epoch = 0
history = model.fit(
    x=g.generator(),
    validation_data=g.generator_validation_data(),
    validation_steps=s2_data.TEST_BATCH_COUNT,
    callbacks=[cb],
    steps_per_epoch=2 * s2_data.TRAIN_BATCH_COUNT,
    epochs=10,
    initial_epoch=initial_epoch,
)
with open("history.pickle", "wb") as file:
    pickle.dump(history.history, file)
