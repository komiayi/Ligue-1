import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Ligue 1 Match Predictor", layout="centered")
# --- LOGO MAPPING FUNCTION ---
def get_github_logo(team_name):
    """Fetches the raw image URL from LuukS/foot-logos repository."""
    base_url = "https://raw.githubusercontent.com/luukhopman/football-logos/master/logos/France%20-%20Ligue%201/"
    # Mapping CSV Team Names -> GitHub File Names
    # Update the left side to match your CSV exactly
    mapping = {
    "Paris SG": "Paris Saint-Germain",
    "Marseille": "Olympique Marseille",
    "Lyon": "Olympique Lyon",
    "Monaco": "AS Monaco",
    "Lille": "LOSC Lille",
    "Lens": "RC Lens",
    "Rennes": "Stade Rennais FC",
    "Nice": "OGC Nice",
    "Reims": "Stade de Reims",
    "Strasbourg": "RC Strasbourg Alsace",
    "Montpellier": "Montpellier HSC",
    "Toulouse": "Toulouse FC",
    "Lorient": "FC Lorient",
    "Nantes": "FC Nantes",
    "Le Havre": "Le Havre AC",
    "Brest": "Stade Brestois 29",
    "Clermont": "Clermont Foot 63",
    "Metz": "FC Metz",
    "Auxerre": "AJ Auxerre",
    "Angers": "Angers SCO",
    "St Etienne": "AS Saint-Étienne",
    "Paris FC": "Paris FC"
}
    file_name = mapping.get(team_name, team_name)
    nom_fichier_url = file_name.replace(" ", "%20")
    return f"{base_url}{nom_fichier_url}.png"

# --- ASSETS LOADING ---
@st.cache_resource
def load_assets():
  # Paths adjusted for your GitHub structure
  model_path = os.path.join('models', 'football_model.pkl')
  stats_path = os.path.join('data', 'latest_team_stats.csv')

  if not os.path.exists(model_path) or not os.path.exists(stats_path):
      st.error("Model or Statistics files not found in the expected directories.")
      st.stop()
      
  model = joblib.load(model_path)
  stats = pd.read_csv(stats_path)
  return model, stats

model, df_stats = load_assets()

# --- SIDEBAR - PROJECT INFO ---
st.sidebar.title("Model insights")
st.sidebar.markdown("""
This model predicts French **Ligue 1** match outcomes based on:
* Expected Goals (xG)
* Conversion Rates
* Historical Team Performance
* Betting Odds (Market Sentiment)
""")

# Display Last Update Date
try:
    timestamp = os.path.getmtime(os.path.join('data', 'latest_team_stats.csv'))
    last_update = datetime.datetime.fromtimestamp(timestamp)
    st.sidebar.info(f" **Last Data Update:** \n{last_update.strftime('%Y-%m-%d %H:%M')}")
except:
    pass

# --- MAIN INTERFACE ---
st.title("Ligue 1 Match Analyzer")
st.markdown("Enter match details below to generate AI-powered probabilities.")

col1, col2 = st.columns(2)
teams = sorted(df_stats['Team'].unique())

with col1:
    home_team = st.selectbox("Select Home Team", options=teams, key='h_team', label_visibility="collapsed")
    logo_home = get_github_logo(home_team)
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="{logo_home}" width="100">
            <p style="font-size: 20px; font-weight: bold; margin-top: 10px;">{home_team}</p>
        </div>
    """, unsafe_allow_html=True)
    
with col2:
    away_team = st.selectbox("Select Away Team", options=teams, key='a_team', label_visibility="collapsed")
    logo_away = get_github_logo(away_team)
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="{logo_home}" width="100">
            <p style="font-size: 20px; font-weight: bold; margin-top: 10px;">{away_team}</p>
        </div>
    """, unsafe_allow_html=True)

# # INPUT FORM
with st.form(key='match_form'):
    st.subheader("Match configuration")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        odd_h = st.number_input("Home odds", value=2.00, format="%.2f")
    with c2:
        odd_d = st.number_input("Draw odds", value=3.20, format="%.2f")
    with c3:
        odd_a = st.number_input("Away odds", value=3.00, format="%.2f")
    st.markdown(" ")
    #odd_d = st.number_input("Draw odds", value=3.20, step=0.05, format="%.2f")

    submit_button = st.form_submit_button(label='RUN PREDICTION', help="Click to predict!")

# PREDICTION LOGIC
if submit_button:
    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        try:
            # Data Extraction
            stats_h = df_stats[df_stats['Team'] == home_team].iloc[0]
            stats_a = df_stats[df_stats['Team'] == away_team].iloc[0]

            # Feature Engineering (must match your training pipeline)
            input_data = pd.DataFrame([{
                'xG_Spec_H': stats_h['HST_mean_h'] * stats_h['conv_mean_h'],
                'xG_Spec_A': stats_a['AST_mean_a'] * stats_a['conv_mean_a'],
                'xGA_Spec_H': stats_h['HST_allowed_mean_h'] * stats_h['conv_allowed_mean_h'],
                'xGA_Spec_A': stats_a['AST_allowed_mean_a'] * stats_a['conv_allowed_mean_a'],
                'Is_start_season': 0,
                'B365H': float(odd_h), 'B365D': float(odd_d), 'B365A': float(odd_a)
            }])

            # Inference
            probs = model.predict_proba(input_data)[0]

            # RESULTS DISPLAY
            st.markdown("Prediction Results")
            st.divider()

            res_h, res_n, res_a = st.columns(3)

            # Using Metrics for a professional dashboard look
            res_h.metric(label=f"{home_team}", value=f"{probs[0]*100:.1f}%")
            res_n.metric(label="Draw", value=f"{probs[1]*100:.1f}%")
            res_a.metric(label=f"{away_team}", value=f"{probs[2]*100:.1f}%")

            # Confidence bar
            highest_prob = max(probs)
            st.write(f"**Model Confidence:**")
            st.progress(float(highest_prob))
            
            st.success("Analysis completed successfully.")
        except ValueError:
           st.error(F"An error occurred during prediction: {e}")

# FOOTER
st.markdown("---")
st.caption("Powered by Scikit-Learn | Data updated weekly from official sources.")
