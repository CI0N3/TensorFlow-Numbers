
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

numbers_mnist = tf.keras.datasets.mnist

(training_pixels, training_labels), (testing_pixels, testing_labels) = numbers_mnist.load_data()

training_pixels = training_pixels / 255.0
testing_pixels = testing_pixels / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(training_pixels, training_labels, epochs=1)

testing_loss, testing_accuracy = model.evaluate(testing_pixels, testing_labels, verbose=2)

for i in range(int(input("How many tests? "))):
    probability_arr = model.predict(np.array([testing_pixels[i]]))
    plt.imshow(testing_pixels[i])
    plt.xlabel(["Prediction: " + str(probability_arr.argmax()),
               "Confidence: " + str(max(np.asarray(probability_arr)[0])),
               "Actual: " + str(testing_labels[i])])
    plt.show()
