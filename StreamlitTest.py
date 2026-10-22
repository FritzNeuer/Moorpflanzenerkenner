import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

MODEL_PFAD = r"G:\adrian\Schule\SF\Moorpflanzen_Modell_Final.keras"
BILD_GROESSE = (180, 180)

class_names = [
        'Besenheide (Calluna vulgaris)',
        'Scheidiges Wollgras (Eriophorum vaginatum)',
        'Sonnentau (Drosera)',
        'Torfmoos (Sphagnum)'
    ]

pflanzen_text = {
        "Besenheide (Calluna vulgaris)": "Die Besenheide (Calluna vulgaris) ist ein Heidekraut mit kleinen, meist rosa bis violetten Blüten. Sie wächst vor allem auf den etwas trockeneren Bulten und in degenerierten Moorflächen, wo sie zahlreichen Insekten (z.B. Bienen oder Schmetterlinge) eine wichtige Nahrungsquelle anbietet. Für mehr Infos siehe https://de.wikipedia.org/wiki/Besenheide",
        "Scheidiges Wollgras (Eriophorum vaginatum)": "Das Scheidige Wollgras (Eriophorum vaginatum) ist ein horstbildendes Sauergras mit charakteristischen weißen Wollschöpfen. Es wächst vor allem auf den torfmoosreichen Bulten intakter und regenerierender Hochmoore, wie sie im Niedersächsischen Hochmoor vorkommen, und trägt dort wesentlich zur Torfbildung bei. Für mehr Infos siehe https://de.wikipedia.org/wiki/Scheiden-Wollgras",
        "Sonnentau (Drosera)": "Der Rundblättrige Sonnentau (Drosera rotundifolia) ist eine kleine, fleischfressende Pflanze mit rundlichen Blättern, die von klebrigen, rötlichen Tentakeln besetzt sind. Er wächst vor allem in den nährstoffarmen Torfmoospolstern und Schlenken intakter und regenerierender Hochmoore, wie im Niedersächsischen Hochmoor, und deckt seinen Stickstoffbedarf durch gefangene Insekten. Für mehr Infos siehe https://de.wikipedia.org/wiki/Rundbl%C3%A4ttriger_Sonnentau",
        "Torfmoos (Sphagnum)": "Das Torfmoos (Sphagnum) ist ein Polster bildendes Moos, das große Mengen Wasser speichern kann und den Boden stark versauert. Es wächst flächendeckend in den nährstoffarmen, nassen Bereichen intakter und regenerierender Hochmoore, wie dem Niedersächsischen Hochmoor, und bildet dort den entscheidenden Baustein für die Torfbildung. Für mehr Infos siehe https://de.wikipedia.org/wiki/Torfmoose"
    }

model = keras.models.load_model(MODEL_PFAD)
print("Modell geladen!")

def bild_erkennen(bild):
    BILD_PFAD = bild

    img = Image.fromarray(bild.astype('uint8'))  # NumPy-Array → PIL-Image
    img = img.resize((180, 180))  # auf 180×180 bringen
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)   # Batch-Dimension

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = f"{100 * np.max(score):.1f}%"
    infotext = pflanzen_text[predicted_class]

    return predicted_class, confidence, infotext

st.title("Moorpflanzen-Erkennung")
st.write("Lade ein Foto einer Moorpflanze hoch und lass sie von der KI bestimmen.")

uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Hochgeladenes Bild", use_container_width=True)

    # Bild vorbereiten
    img_resized = img.resize(BILD_GROESSE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Vorhersage
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = f"{100 * np.max(score):.1f}%"
    infotext = pflanzen_text[predicted_class]

    # Ergebnisse anzeigen
    st.subheader("Ergebnis")
    st.write(f"**Erkannte Pflanze:** {predicted_class}")
    st.write(f"**Sicherheit:** {confidence}")
    st.write(f"**Informationen:** {infotext}")