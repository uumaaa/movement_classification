import tensorflow as tf

n_timesteps = 100
features = 224
input1 = tf.keras.layers.Input(shape=(n_timesteps, features))
lstm = tf.keras.layers.LSTM(units=100, activation="relu", return_sequences=False)(
    input1
)
outputs = tf.keras.layers.Dense(n_timesteps, activation="sigmoid")(lstm)
model = tf.keras.Model(inputs=input1, outputs=outputs)

x = tf.random.normal((1, n_timesteps, features))
y = tf.random.uniform((1, n_timesteps), dtype=tf.int32, maxval=2)

print(x)
print(y)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(x, y, epochs=6)
