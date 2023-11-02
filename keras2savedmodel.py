import tensorflow as tf

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
model.load_weights("weight.hdf5")
tf.saved_model.save(model, "data/04_s2_savedmodel/")
