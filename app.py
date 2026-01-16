import streamlit as st
import pandas as pd
import joblib
import os

# Configuration
st.set_page_config(page_title="Ligue 1 Predictor", layout="centered")

# Chargement
@st.cache_resource
def load_assets():
  model_path = os.path.join(models, 'football_model.pkl')
  stats_path = os.path.join(data, 'latest_team_stats.csv')

  # Vérification de l'existence des fichiers
  if not os.path.exists(model_path) or not os.path.exists(stats_path):
      st.error(f"Erreur : Fichiers introuvables. Vérifiez que {model_path} et {stats_path} existent.")
      st.stop()
      
  model = joblib.load(model_path)
  stats = pd.read_csv(stats_path)
  return model, stats

model, df_stats = load_assets()

st.title("Analyseur de matchs Ligue 1")

# LE FORMULAIRE : C'est lui qui empêche le scintillement
with st.form(key='mon_formulaire1'):
    #
    st.subheader("Configuration du match")

    col1, col2 = st.columns(2)

    with col1:
        #
        equipes = sorted(df_stats['Team'].unique())
        home_team = st.selectbox("Équipe domicile", options=equipes, key='h_team')
        odd_h = st.number_input(label = "Cote victoire domicile", value = 2.00, step=0.05, format="%.2f")
        #odd_h = st.text_input("Cote Victoire Domicile", value="2.00")

    with col2:
        away_team = st.selectbox("Équipe extérieur", options=equipes, key='a_team')
        odd_a = st.number_input(label = "Cote victoire extérieur", value = 3.00, step=0.05, format="%.2f")
        #odd_a = st.text_input("Cote Victoire Extérieur", value="3.00")

    odd_d = st.number_input("Cote match nul", value=3.20, step=0.05, format="%.2f")
    #odd_d = st.text_input("Cote Match Nul", value="3.20")
    # Le bouton d'envoi unique
    submit_button = st.form_submit_button(label='LANCER LA PRÉDICTION', help="Click to predict!")

# Le calcul ne se lance QUE si on clique sur le bouton
if submit_button:
    #
    try:
        #
        odd_h = float(odd_h)
        odd_d = float(odd_d)
        odd_a = float(odd_a)

        if home_team == away_team:
            st.error("Veuillez choisir deux équipes différentes.")
        else:
            stats_h = df_stats[df_stats['Team'] == home_team].iloc[0]
            stats_a = df_stats[df_stats['Team'] == away_team].iloc[0]

            # Calcul des xG spécifiques pour l'entrée du modèle
            input_data = pd.DataFrame([{
                'xG_Spec_H': stats_h['HST_mean_h'] * stats_h['conv_mean_h'],
                'xG_Spec_A': stats_a['AST_mean_a'] * stats_a['conv_mean_a'],
                'xGA_Spec_H': stats_h['HST_allowed_mean_h'] * stats_h['conv_allowed_mean_h'],
                'xGA_Spec_A': stats_a['AST_allowed_mean_a'] * stats_a['conv_allowed_mean_a'],
                'Is_start_season': 0,
                'B365H': odd_h, 'B365D': odd_d, 'B365A': odd_a
            }])
            probs = model.predict_proba(input_data)[0]

            st.success("Analyse terminée !")
            st.divider()

            res_h, res_n, res_a = st.columns(3)

            with res_h:
                st.markdown(f"Victoire {home_team}")
                st.info(f"{probs[0]*100:.1f}%")

            with res_n:
                st.markdown("Match Nul")
                st.info(f"{probs[1]*100:.1f}%")

            with res_a:
                st.markdown(f"Victoire {away_team}")
                st.info(f"{probs[2]*100:.1f}%")
    except ValueError:
        st.error("Vérifiez le format des cotes (utilisez un point pour les décimales, ex: 1.50)")

