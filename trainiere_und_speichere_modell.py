# trainiere_und_speichere_modell.py

# === Notwendige Importe ===
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Für den optionalen Test nach dem Training
import cv2           # Für den optionalen Test nach dem Training

print("--- Start: KI zum Zahlenerkennen - Training ---")

# === Schritt 2: Daten laden und vorbereiten ===
# ... (Dein Code für Schritt 2 bleibt unverändert) ...
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
print("   Pixelwerte auf den Bereich 0 bis 1 normalisiert.")


# === Schritt 3: Modell bauen ===
# ... (Dein Code für Schritt 3 bleibt unverändert) ...
input_shape = (28, 28)
model = Sequential(name="Zahlenmodell_V2")
model.add(Input(shape=input_shape, name="Eingabe"))
model.add(Flatten(name="Flatten"))
model.add(Dense(128, activation='relu', name="Versteckt"))
model.add(Dropout(0.2, name="Dropout"))
model.add(Dense(10, activation='softmax', name="Ausgabe"))
model.summary()

# === Schritt 4: Modell kompilieren und trainieren ===
# ... (Dein Code für Schritt 4 bleibt unverändert) ...
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("\nStarte das Training...")
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_split=0.1
                   )
print("Training abgeschlossen!")

# === Schritt 5: Modell bewerten ===
# ... (Dein Code für Schritt 5 bleibt unverändert) ...
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"   Genauigkeit auf ungesehenen Testdaten: {test_acc*100:.2f}%")

# === SCHRITT ZUM SPEICHERN DES TRAINIERTEN MODELLS ===
modell_speicherpfad = 'mein_zahlen_erkennungs_modell.keras'
print(f"\n--- Speichere trainiertes Modell unter: {modell_speicherpfad} ---")
try:
    model.save(modell_speicherpfad)
    print(f"   Modell erfolgreich gespeichert!")
except Exception as e:
    print(f"FEHLER beim Speichern des Modells: {e}")


# === Optional: Teste das GERADE TRAINIERTE Modell mit einem eigenen Bild ===
# Die Funktion vorhersage_eigenes_bild muss hier definiert sein (oder importiert werden)
def vorhersage_eigenes_bild(bild_pfad, model_zum_testen):
    try:
        img = Image.open(bild_pfad).convert('L')
        img_cv = np.array(img)
        img_resized_cv = cv2.resize(img_cv, (28, 28), interpolation=cv2.INTER_AREA)
        # WICHTIG: Vorverarbeitung anpassen!
        # Beispiel: WENN deine Bilder schwarze Ziffern auf weißem Grund haben:
        img_resized_cv = 255 - img_resized_cv
        img_normalized = img_resized_cv / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        prediction_probabilities = model_zum_testen.predict(img_batch)
        predicted_digit = np.argmax(prediction_probabilities[0])
        print(f"\nVorhersage für '{bild_pfad}': {predicted_digit} (Wahrscheinlichkeiten: {prediction_probabilities[0]})")
        plt.figure(figsize=(3,3))
        plt.imshow(img_normalized, cmap='gray')
        plt.title(f"Test: {predicted_digit}")
        plt.show()
        return predicted_digit
    except Exception as e:
        print(f"Fehler bei eigener Vorhersage: {e}")
        return None

print("\n--- Trainings-Skript beendet ---")