
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoders
try:
    model = joblib.load('modelo_naive_bayes.pkl')
    # Recreate the label encoders and fit them with the original data categories
    # This assumes you have the original data categories available or saved separately
    # For this example, I'll re-use the categories from the previous notebook
    data = {
        "Horas de Estudio": ["Alta", "Baja", "Baja", "Alta", "Alta"],
        "Asistencia": ["Buena", "Buena", "Mala", "Mala", "Buena"],
        "Resultado": ["Sí", "No", "No", "Sí", "Sí"]
    }
    df_temp = pd.DataFrame(data)

    label_encoders = {}
    for column in ["Horas de Estudio", "Asistencia", "Resultado"]:
        label_encoders[column] = LabelEncoder()
        label_encoders[column].fit(df_temp[column])

except FileNotFoundError:
    st.error("Error: El archivo del modelo 'modelo_naive_bayes.pkl' no fue encontrado. Asegúrate de haber ejecutado la parte de entrenamiento en Google Colab primero.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo o los codificadores: {e}")
    st.stop()


# Streamlit app
st.title("Predicción de Clase")
st.markdown("<h2 style='color: red;'>Elaborado por: Klaudialiliana</h2>", unsafe_allow_html=True)

st.write("Selecciona los valores para las entradas de las variables para realizar la predicción.")

# Input widgets for user to select features
horas_estudio = st.selectbox("Horas de Estudio:", label_encoders["Horas de Estudio"].classes_)
asistencia = st.selectbox("Asistencia:", label_encoders["Asistencia"].classes_)

# Prepare the input data for prediction
new_observation = pd.DataFrame({
    "Horas de Estudio": [horas_estudio],
    "Asistencia": [asistencia]
})

# Encode the new observation using the loaded label encoders
try:
    for column in ["Horas de Estudio", "Asistencia"]:
        new_observation[column] = label_encoders[column].transform(new_observation[column])
except ValueError as e:
    st.error(f"Error al codificar la entrada. Asegúrate de que las opciones seleccionadas son válidas. Detalle: {e}")
    st.stop()


# Make the prediction
predicted_label_encoded = model.predict(new_observation)
predicted_label = label_encoders["Resultado"].inverse_transform(predicted_label_encoded)

# Display the result
st.write("---")
st.subheader("Resultado de la Predicción")

if predicted_label[0] == "Sí":
    st.success(f"🎉 ¡Felicidades, aprueba! ({predicted_label[0]})")
else:
    st.error(f"😔 No aprueba. ({predicted_label[0]})")
