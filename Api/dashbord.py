import pandas as pd
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import shap

st.title("Attribution du crédit")

# Utilisez st.file_uploader pour permettre aux utilisateurs d'importer un fichier CSV
uploaded_file = st.file_uploader("Upload customer file to be verified", type=["csv"])

# Vérifiez si un fichier a été téléchargé
if uploaded_file:
    # Chargez les données depuis le fichier CSV
    ride = pd.read_csv(uploaded_file)
    df_columns=ride.columns

# Utilisez st.text_input pour obtenir l'identifiant client
id = st.text_input("Entrez l'identifiant client:")

# Variable pour stocker le dernier ID traité
last_processed_id = None

# Créer un conteneur vide pour afficher le message
message_container, plot_container = st.columns([10, 2])

# Utilisez st.button pour exécuter l'action une fois que le bouton est pressé
if st.button("Run"):
    try:
        # Vérifier si l'ID est vide ou non numérique
        if not id or not id.isdigit():
            raise ValueError("Veuillez entrer un identifiant client valide (entier).")

        # Convertir l'ID entré en entier
        current_id = int(id)

        # Vérifier si l'ID actuel est différent du dernier ID
        if current_id != last_processed_id:
            ride_subset = ride.iloc[[current_id]]
            ride_data = ride_subset.to_dict(orient='records')

            url = 'http://localhost:9696/predict'
            
        # Afficher un indicateur visuel pendant le chargement
        with st.spinner("Chargement en cours..."):
            response = requests.post(url, json=ride_data)

            response_data = response.json()

            if "prediction" in response_data:
                prediction = response_data["prediction"]
                shap_values_data = response_data["shap_values"]
                feature_globale = response_data["feature_global"]
                probability = response_data["probability"]

                # Afficher le message approprié en fonction de la prédiction avec style personnalisé
                if prediction == 0:
                    message_container.markdown("<p style='color:green; font-size:40px;'>Le crédit sera accordé avec une probabilité de {:.2%}.</p>".format(probability), unsafe_allow_html=True)

                elif prediction == 1:
                    message_container.markdown("<p style='color:red; font-size:40px;'>Le crédit ne sera pas accordé avec une probabilité de {:.2%}.</p>".format(probability), unsafe_allow_html=True)

                # Afficher les valeurs SHAP (exemple)
                exp = shap.Explanation(np.array(shap_values_data['values']),np.array(shap_values_data['base_values']), df_columns)
                fig, ax = plt.subplots(figsize=(20, 20))
                shap.plots.waterfall(exp[0], max_display=30, show=True)
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(20, 20))
                sns.barplot(x="importance", y="feature", data=pd.DataFrame(response_data["feature_global"]))
                st.pyplot(fig)

                # Effacer automatiquement l'identifiant actuel après avoir traité l'appel API
                id = ""

                # Mettre à jour le dernier ID traité
                last_processed_id = current_id
            else:
                st.warning("La clé 'prediction' est absente dans la réponse de l'API. Veuillez vérifier les résultats.")
    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error("Erreur lors de la communication avec l'API. Veuillez réessayer.")