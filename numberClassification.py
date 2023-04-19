%matplotlib inline

import numpy as np
import matplotlib.pylab as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist

# Assume 'a' is an activation map (output of a convolution with a kernel)
a = np.random.random(size=(10, 12, 12, 1))

activation_map = keras.backend.variable(a)

max_pool_2D(activation_map)

# Let us plot the output
fig, ax = plt.subplots(figsize=[10,10])
im = ax.imshow(activation_map[0,:,:,0], cmap='gray', vmin=0, vmax=1)


for i in range(12):
    for j in range(12):
        text = ax.text(j, i, np.format_float_positional(activation_map[0,i,j,0], precision=3),
                       ha="center", va="center", color="blue")

max_pool_2D = keras.layers.MaxPool2D(
    pool_size=(2, 2),
    strides=2,
    padding='valid',
    data_format=None,
)

pooled_map = max_pool_2D(activation_map)

fig, ax = plt.subplots(figsize=[10, 10])
im = ax.imshow(pooled_map[0,:,:,0], cmap='gray', vmin=0, vmax=1)

for i in range(6):
    for j in range(6):
        text = ax.text(j, i, np.format_float_positional(pooled_map[0,i,j,0], precision=3),
                       ha="center", va="center", color="blue")  
                       
model = Sequential([
    Conv2D(filters=10, kernel_size=3, strides=(1, 1), padding='valid'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(10)
])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data so it has a channel dimension
rows, cols = x_train.shape[-2:]
x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)

# Convert pixel intensities to values between 0 and 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
    
# Convert target vectors to one-hot encoding
num_classes = len(set(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

mini_batch_size = 10

net = Sequential([ 
        keras.Input(shape=(28, 28, 1)),
        Conv2D(28, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(56, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(10, activation="softmax")])

net.summary()

batch_size = 128
epochs = 15

net.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

