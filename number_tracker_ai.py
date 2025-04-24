# Importiere die nötigen Werkzeuge


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Lade den MNIST-Datensatz
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Zeige das erste Bild aus dem Trainingsdatensatz
plt.imshow(x_train[0], cmap='gray') # cmap='gray' für Schwarz-Weiß
plt.title(f"Dieses Bild zeigt die Ziffer: {y_train[0]}")
plt.show()