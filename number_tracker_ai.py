# Importiere die nötigen Werkzeuge


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Lade den MNIST-Datensatz
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

input_shape = (28, 28)
print(f"\n1. Erwartete Form der Eingabedaten (ein Bild): {input_shape}")

print("\n2. Erstelle einen leeren 'Sequential'-Behälter für die Schichten.")
model = Sequential(name="Zeichencheck") # Geben wir ihm einen Namen
print(f"   Leeres Modell '{model.name}' wurde erstellt.")

print(f"\n3. Füge die 'Flatten'-Schicht hinzu.")
print(f"   Zweck: Wandelt die 2D-Bildstruktur {input_shape} in eine 1D-Liste um.")
# Wir berechnen die resultierende Länge: 28 * 28 = 784
flattened_size = np.prod(input_shape)

model.add(Input(shape=input_shape, name="Eingabe_Schicht")) # Gib der Schicht einen Namen
print(f"   'Input'-Schicht hinzugefügt. Erwartete Eingabeform: (None, {input_shape[0]}, {input_shape[1]})")
print(model, "ajsdaklsd")
print(f"   'Flatten'-Schicht hinzugefügt. Aktuelle Datenform im Modell: (None, {flattened_size})")

print("Hello World")