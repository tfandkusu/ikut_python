import tensorflow as tf

mnist = tf.keras.datasets.mnist
# データセットを読み込む
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# 機械学習モデルを構築する
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)
# モデルのコンパイル
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
# モデルのトレーニング
model.fit(x_train, y_train, epochs=5)
# モデルの評価
model.evaluate(x_test, y_test, verbose=2)
