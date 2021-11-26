import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

input_shape = (26,26,1)
num_classes=10

(x_train, y_train) , (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

model = tf.keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ]
)

model.summary()


model.compile(loss = tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

callback = [tf.keras.callbacks.ModelCheckpoint('saved_model.h5')]
model.fit(x_train, y_train, batch_size=32, epochs=2, validation_split=0.2, callbacks=callback)

score = model.evaluate(x_test, y_test, verbose=0)

print(f'test loss if {score[0]*100}%')
print(f'test accuracy is {score[1]*100}%')


































