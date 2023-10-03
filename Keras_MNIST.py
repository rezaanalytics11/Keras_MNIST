import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import Normalizer
import sklearn
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras import datasets
from keras import layers



num_classes = 10
input_shape = (28, 28, 1)

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()



x_train=x_train/255
x_test=x_test/255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)



y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_test[0].shape)

for i in range(400):
    plt.subplot(20,20,i+1)
    plt.imshow(x_train[i])
    plt.axis('off')
plt.show()

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['loss'],label='loss')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(['accuracy','loss'])
plt.show()