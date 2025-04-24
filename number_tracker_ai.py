# Importiere die nötigen Werkzeuge


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Lade den MNIST-Datensatz
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# Baue das Modell Schicht für Schicht
model = tf.keras.models.Sequential([
  # 1. Schicht: "Flatten" - Macht aus dem 28x28 Bild eine flache Liste von 784 Pixeln
  tf.keras.layers.Flatten(input_shape=(28, 28)),

  # 2. Schicht: "Dense" - Eine voll vernetzte Schicht mit 128 Neuronen
  # 'relu' ist eine Aktivierungsfunktion - hilft dem Netz, komplexe Muster zu lernen
  tf.keras.layers.Dense(128, activation='relu'),

  # 3. Schicht: "Dropout" - Eine Technik, um Überanpassung zu vermeiden (optional, aber gut)
  # Schaltet zufällig Neuronen während des Trainings aus
  tf.keras.layers.Dropout(0.2),

  # 4. Schicht: "Dense" - Die Ausgabeschicht. Sie muss 10 Neuronen haben (für Ziffern 0-9).
  # 'softmax' sorgt dafür, dass die Ausgabe Wahrscheinlichkeiten sind (z.B. 80% sicher, dass es eine 7 ist)
  tf.keras.layers.Dense(10, activation='softmax')
])

print(model)
print("Hello World")