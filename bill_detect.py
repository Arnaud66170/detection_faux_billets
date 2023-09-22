import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def bill_detect(df, model_scaler_dict, accuracy):
    model = model_scaler_dict['model']
    scaler = model_scaler_dict['scaler']

    selected_columns = ["length", "margin_low", "margin_up"]
    df_num = df[selected_columns]

    sample_scaled = scaler.transform(df_num)

    predict = model.predict(sample_scaled)

    df["predict"] = predict

    st.subheader("Résultats de la détection des billets :")
    st.dataframe(df)
    st.bar_chart(df["predict"].value_counts())
    # Précision du modèle
    st.subheader(f"Taux de précision du modèle : {accuracy:.2f}%")

st.title("Détection de faux billets")
st.markdown("Cette application vérifie l'authenticité des billets bancaires en fonction de leurs caractéristiques, par régression logstique.")

uploaded_file = st.file_uploader("Chargement fichier CSV")

if uploaded_file is not None:
    try:
        model_scaler_dict = joblib.load("reg_log.pkl")  # Adjusted path
        accuracy = 98.67  # The accuracy of the model
        df = pd.read_csv(uploaded_file, sep =",")
        bill_detect(df, model_scaler_dict, accuracy)
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement du modèle : {e}")
