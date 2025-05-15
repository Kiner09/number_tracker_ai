# lade_und_benutze_modell.py

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Die Funktion vorhersage_eigenes_bild muss hier definiert sein (oder importiert werden)
# Sie MUSS die gleiche Vorverarbeitung machen wie beim Training/ersten Test!
def vorhersage_eigenes_bild(bild_pfad, model_zum_testen):
    try:
        img = Image.open(bild_pfad).convert('L')
        print(f"Eigenes Bild geladen: {bild_pfad}")
        img_cv = np.array(img)
        img_resized_cv = cv2.resize(img_cv, (28, 28), interpolation=cv2.INTER_AREA)

        # WICHTIG: Vorverarbeitung anpassen! Diese muss konsistent sein!
        # Wenn du z.B. im Trainingsskript nach dem Training für deine Testbilder
        # invertiert hast (weil sie schwarz auf weiß waren), musst du das hier auch tun.
        # Wenn deine Bilder schon wie MNIST sind (weiß auf schwarz), brauchst du keine Invertierung.
        # Beispiel: WENN deine Bilder schwarze Ziffern auf weißem Grund haben:
        img_resized_cv = 255 - img_resized_cv
        print("Info: Bildfarben werden ggf. invertiert.")

        img_normalized = img_resized_cv / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        print(f"   Vorbereitetes Bild Shape für Modell: {img_batch.shape}")

        prediction_probabilities = model_zum_testen.predict(img_batch)
        predicted_digit = np.argmax(prediction_probabilities[0])

        print(f"   Vorhersage-Wahrscheinlichkeiten: {prediction_probabilities[0]}")
        print(f"   Das Modell sagt vorher: {predicted_digit}")

        plt.figure(figsize=(6,3))
        plt.subplot(1, 2, 1)
        original_img_for_display = Image.open(bild_pfad)
        plt.imshow(original_img_for_display)
        plt.title("Originalbild")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img_normalized, cmap='gray')
        plt.title(f"Vorbereitet & Vorhersage: {predicted_digit}")
        plt.axis('off')
        plt.show()
        return predicted_digit
    except FileNotFoundError:
        print(f"FEHLER: Datei nicht gefunden unter {bild_pfad}")
        return None
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None

print("--- Starte Programm zur Nutzung des trainierten KI-Modells ---")

# === Modell laden ===
modell_speicherpfad = 'mein_zahlen_erkennungs_modell.keras' # Derselbe Pfad wie beim Speichern!
print(f"\n--- Lade gespeichertes Modell von: {modell_speicherpfad} ---")

try:
    geladenes_modell = tf.keras.models.load_model(modell_speicherpfad)
    print("   Modell erfolgreich geladen!")
    geladenes_modell.summary() # Zeigt die Struktur des geladenen Modells

    # === Geladenes Modell für Vorhersagen verwenden ===
    eigener_bild_pfad1 = r'C:\Users\admin\Downloads\1_28.png' # Euer Bild
    # eigener_bild_pfad2 = r'C:\pfad\zu\deiner_ziffer_3.png' # Ein anderes Testbild

    print("\n--- Teste Vorhersage mit geladenem Modell ---")
    if eigener_bild_pfad1:
        vorhergesagte_ziffer = vorhersage_eigenes_bild(eigener_bild_pfad1, geladenes_modell)
        if vorhergesagte_ziffer is not None:
            print(f"Die KI hat die Ziffer {vorhergesagte_ziffer} auf deinem Bild '{eigener_bild_pfad1}' erkannt.")

    # if eigener_bild_pfad2:
    #     vorhergesagte_ziffer_2 = vorhersage_eigenes_bild(eigener_bild_pfad2, geladenes_modell)
    #     if vorhergesagte_ziffer_2 is not None:
    #         print(f"Die KI hat die Ziffer {vorhergesagte_ziffer_2} auf deinem Bild '{eigener_bild_pfad2}' erkannt.")

except FileNotFoundError:
    print(f"FEHLER: Modelldatei '{modell_speicherpfad}' nicht gefunden. Bitte zuerst das Skript 'trainiere_und_speichere_modell.py' ausführen.")
except Exception as e:
    print(f"FEHLER beim Laden oder Verwenden des Modells: {e}")

print("\n--- Programm beendet ---")