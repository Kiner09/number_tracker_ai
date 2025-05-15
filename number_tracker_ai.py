# === Notwendige Importe ===
import tensorflow as tf
from tensorflow.keras.models import Sequential
# Input Layer importieren für die beste Praxis
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
import numpy as np
# (Optional, aber gut für spätere Visualisierung)
import matplotlib.pyplot as plt

print("--- Start: KI zum Zahlenerkennen ---")

# === Schritt 2: Daten laden und vorbereiten ===
print("\n--- Schritt 2: Lade und bereite MNIST Daten vor ---")
# Lade den MNIST-Datensatz (Trainings- und Testdaten)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(f"   Daten geladen: {x_train.shape[0]} Trainingsbilder, {x_test.shape[0]} Testbilder.")
print(f"   Bildformat: {x_train.shape[1:]} Pixel")

# Daten vorbereiten (Normalisierung: Pixelwerte von 0-255 auf 0-1 bringen)
x_train = x_train / 255.0
x_test = x_test / 255.0
print("   Pixelwerte auf den Bereich 0 bis 1 normalisiert.")

# === Schritt 3: Modell bauen (mit kürzeren Prints & Input Layer) ===
print("\n--- Schritt 3: Baue das Neuronale Netz ---")

# Eingabeform definieren (wichtig für die erste Schicht)
input_shape = (28, 28)
flattened_size = np.prod(input_shape) # 28*28 = 784
num_hidden_neurons = 128
num_output_neurons = 10
dropout_rate = 0.2

print(f"1. Eingabeform definiert: {input_shape}")

# Modell erstellen
model = Sequential(name="Zahlenmodell_V2")
print(f"2. Sequential Modell '{model.name}' erstellt.")

# Input Layer hinzufügen (Best Practice)
model.add(Input(shape=input_shape, name="Eingabe"))
print(f"3. Input Layer hinzugefügt (erwartet {input_shape}).")

# Flatten Layer hinzufügen
model.add(Flatten(name="Flatten"))
print(f"4. Flatten Layer hinzugefügt (wandelt Bild -> {flattened_size} Neuronen).")

# Dense Layer (versteckt) hinzufügen
model.add(Dense(num_hidden_neurons, activation='relu', name="Versteckt"))
print(f"5. Dense Layer (versteckt) hinzugefügt ({num_hidden_neurons} Neuronen, ReLU).")

# Dropout Layer hinzufügen
model.add(Dropout(dropout_rate, name="Dropout"))
print(f"6. Dropout Layer hinzugefügt (Rate: {dropout_rate*100}%).")

# Dense Layer (Ausgabe) hinzufügen
model.add(Dense(num_output_neurons, activation='softmax', name="Ausgabe"))
print(f"7. Dense Layer (Ausgabe) hinzugefügt ({num_output_neurons} Neuronen, Softmax).")

# Modell-Zusammenfassung anzeigen
print("\nModell Zusammenfassung:")
model.summary()

# === Schritt 4: Modell kompilieren und trainieren ===
print("\n--- Schritt 4: Kompiliere und trainiere das Modell ---")

# Kompilieren: Optimizer, Verlustfunktion und Metriken festlegen
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("   Modell kompiliert (Optimizer: adam, Loss: sparse_categorical_crossentropy).")

# Trainieren: Modell lernt von den Trainingsdaten
print("\nStarte das Training...")
# Wir speichern den Verlauf, falls wir ihn später analysieren wollen
history = model.fit(x_train, y_train,
                    epochs=5,           # Anzahl der Trainingsdurchläufe
                    batch_size=32,      # Wie viele Bilder pro Schritt verarbeitet werden (optional, Standard ist 32)
                    validation_split=0.1 # Optional: Nutze 10% der Trainingsdaten zur Validierung während des Trainings
                   )
print("Training abgeschlossen!")

# (Optional: Schritt 5 - Bewertung hinzufügen)
print("\n--- Schritt 5: Bewerte das Modell ---")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0) # verbose=0 für weniger Output
print(f"   Genauigkeit auf ungesehenen Testdaten: {test_acc*100:.2f}%")

print("\n--- Fertig ---")