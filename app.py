import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Ligue 1 Match Predictor",
    layout="centered"
)

# --- LOGO MAPPING FUNCTION ---
def get_github_logo(team_name):
    """Fetches the raw image URL from luukhopman/foot-logos repository."""
    base_url = "https://raw.githubusercontent.com/luukhopman/football-logos/master/logos/France%20-%20Ligue%201/"

    # URL d'un ballon de foot générique (ou logo L1) en cas d'absence
    default_logo = "https://cdn-icons-png.flaticon.com/512/33/33736.png"
    
    # Mapping CSV Team Names -> GitHub File Names
    # Update the left side to match your CSV exactly
    mapping = {
        #
        "Paris SG": "Paris Saint-Germain",
        "Marseille": "Olympique Marseille",
        "Lyon": "Olympique Lyon",
        "Monaco": "AS Monaco",
        "Lille": "LOSC Lille",
        "Lens": "RC Lens",
        "Rennes": "Stade Rennais FC",
        "Nice": "OGC Nice",
        "Strasbourg": "RC Strasbourg Alsace",
        "Lorient": "FC Lorient",
        "Nantes": "FC Nantes",
        "Le Havre": "Le Havre AC",
        "Brest": "Stade Brestois 29",
        "Metz": "FC Metz",
        "Auxerre": "AJ Auxerre",
        "Angers": "Angers SCO",
        "Paris FC": "Paris FC",
        "Toulouse": "FC Toulouse"
    }

    if team_name in mapping:
        file_name = mapping[team_name]
        nom_fichier_url = file_name.replace(" ", "%20")
        return f"{base_url}{nom_fichier_url}.png"
    else:
        return default_logo
        
    #file_name = mapping.get(team_name, team_name)
    #nom_fichier_url = file_name.replace(" ", "%20")
    #return f"{base_url}{nom_fichier_url}.png"

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

# Loading assets with a professional touch
with st.spinner("Loading statistical engine..."):
    model, df_stats = load_assets()

# --- SIDEBAR LOGO (Official WebP Version) ---
l1_header_html = """
<div style="
    background-color: #002D72; 
    padding: 20px; 
    border-radius: 15px; 
    text-align: center; 
    border: 2px solid #FFFFFF;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
">
    <img src="https://ligue1.com/images/Logo_Ligue1_large.webp" 
         width="130" 
         style="margin-bottom: 10px;">
</div>
"""

st.sidebar.markdown(l1_header_html, unsafe_allow_html=True)

st.sidebar.title("Model Insights")
st.sidebar.divider()
st.sidebar.markdown("""
This predictor uses a **Scikit-Learn** pipeline to analyze:
* **xG Metrics**
* **Conversion Rates**
* **Team Form**
* **Market Odds**
""")

# Display Last Update Date
try:
    timestamp = os.path.getmtime(os.path.join('data', 'latest_team_stats.csv'))
    last_update = datetime.datetime.fromtimestamp(timestamp)
    st.sidebar.info(f" **Last Data Update:** \n{last_update.strftime('%Y-%m-%d %H:%M')}")
except:
    pass

# --- MAIN INTERFACE ---
st.title("Match Predictor")
st.markdown("""
    <p style='font-size: 1.1em; color: gray;'>
    Statistical analysis & predictions based <b>Scikit-Learn</b> model.
    <br>Season 2025-2026
    </p>
    """, 
    unsafe_allow_html=True
)
st.markdown(" ")

col1, col_vs, col2 = st.columns([4, 2, 4])
mapping_clubs_2026 = [
    "Paris SG", "Marseille", "Lyon", "Monaco", "Lille", "Lens", 
    "Rennes", "Nice", "Strasbourg", "Toulouse", 
    "Lorient", "Nantes", "Le Havre", "Brest", "Auxerre", "Angers", "Paris FC", "Metz"
]

# Filtrer les équipes pour n'afficher que les 18 de la saison 2025-2026
teams = sorted([t for t in df_stats['Team'].unique() if t in mapping_clubs_2026])

with col1:
    home_team = st.selectbox("Select Home Team", options=teams, key='h_team', label_visibility="collapsed")
    logo_home = get_github_logo(home_team)
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="{logo_home}" width="50">
            <p style="font-size: 20px; font-weight: bold; margin-top: 10px;">{home_team}</p>
        </div>
    """, unsafe_allow_html=True)

with col_vs:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
            <h1 style="color: #FF4B4B; font-size: 45px; margin: 0; line-height: 1;">VS</h1>
            <p style="font-size: 12px; color: gray; margin: 0; white-space: nowrap;">Ligue 1 McDonald's</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    away_team = st.selectbox("Select Away Team", options=teams, key='a_team', label_visibility="collapsed")
    logo_away = get_github_logo(away_team)
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="{logo_away}" width="50">
            <p style="font-size: 20px; font-weight: bold; margin-top: 10px;">{away_team}</p>
        </div>
    """, unsafe_allow_html=True)
    
#st.markdown("---")

# --- INPUT FORM ---
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'>Match configuration</h3>", unsafe_allow_html=True)
    with st.form(key='match_form'):
        c1, c2, c3 = st.columns(3)
        with c1:
            odd_h = st.number_input("Home odds", value=2.00, format="%.2f")
        with c2:
            odd_d = st.number_input("Draw odds", value=3.20, format="%.2f")
        with c3:
            odd_a = st.number_input("Away odds", value=3.00, format="%.2f")
            
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
            with st.container(border=True):
                #
                st.markdown("""
                    <style>
                        .result-card {
                            background-color: #f8f9fa;
                            border-radius: 5px;
                            padding: 20px;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                            text-align: center;
                            border: 1px solid #e0e0e0;
                        }
                        .prob-bar-container {
                            display: flex;
                            width: 100%;
                            height: 20px;
                            border-radius: 5px;
                            overflow: hidden;
                            box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
                            margin: 20px 0;
                        }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown("<h2 style='text-align: center;'>Prediction results</h2>", unsafe_allow_html=True)
    
                # 
                p_h, p_d, p_a = probs[0]*100, probs[1]*100, probs[2]*100
                #{p_a:.0f}
                st.markdown(f"""
                    <div class="prob-bar-container">
                        <div style="width: {p_h}%; background: linear-gradient(90deg, #2e7d32, #4caf50); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);"></div>
                        <div style="width: {p_d}%; background: linear-gradient(90deg, #616161, #9e9e9e); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);"></div>
                        <div style="width: {p_a}%; background: linear-gradient(90deg, #c62828, #f44336); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);"></div>
                    </div>
                """, unsafe_allow_html=True)

                # 
                c1, c2, c3 = st.columns(3)

                with c1:
                    #
                    st.markdown(f"""
                        <div class="result-card">
                            <p style="color: #2e7d32; font-weight: bold; margin-bottom: 0;">HOME</p>
                            <h2 style="margin: 0;color: var(--text-color, #010000);">{p_h:.1f}%</h2>
                            <p style="font-size: 0.9em; color: gray;">{home_team}</p>
                        </div>
                    """, unsafe_allow_html=True)

                with c2:
                    st.markdown(f"""
                        <div class="result-card">
                            <p style="color: #616161; font-weight: bold; margin-bottom: 0;">DRAW</p>
                            <h2 style="margin: 0;color: var(--text-color, #010000);">{p_d:.1f}%</h2>
                            <p style="font-size: 0.9em; color: gray;"></p>
                        </div>
                    """, unsafe_allow_html=True)
            
                with c3:
                    st.markdown(f"""
                        <div class="result-card">
                            <p style="color: #c62828; font-weight: bold; margin-bottom: 0;">AWAY</p>
                            <h2 style="margin: 0;color: var(--text-color, #010000);">{p_a:.1f}%</h2>
                            <p style="font-size: 0.9em; color: gray;">{away_team}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
                #
                st.markdown("<br>", unsafe_allow_html=True)
                if p_h > p_a and p_h > p_d:
                    st.info(f"**Model Output:** Statistical edge for **{home_team}** at home.")
                elif p_a > p_h and p_a > p_d:
                    st.info(f"**Model Output:** Statistical edge for **{away_team}** on the road.")
                else:
                    st.info("**Model Output:** High entropy detected. Match expected to be highly competitive (Draw tendency).")
            
                st.success("Analysis completed successfully.")
        except Exception as e:
           st.error(F"An error occurred during prediction: {e}")

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

# Deux colonnes simples : Tech & Data
foot_col1, foot_col2 = st.columns(2)

with foot_col1:
    st.markdown("##### Technology")
    st.caption("Developed with **Python** & **Scikit-Learn**")
    st.caption("Interface by **Streamlit**")

with foot_col2:
    st.markdown("##### Data Status")
    st.caption("Sources: Official Ligue 1 Statistics")
    st.caption("Last Update: Weekly Sync")

# Disclaimer discret et centré
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.75em; margin-top: 30px;'>"
    "For statistical purposes only. Past performance does not guarantee future results."
    "</p>", 
    unsafe_allow_html=True
)
