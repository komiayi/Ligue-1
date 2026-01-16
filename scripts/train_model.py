import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# ==========================================
# 1. FONCTION DE PRÉPARATION (FEATURE ENGINEERING)
# ==========================================
def data_preparation(df):
    """
    Transforms raw match data into engineered features (xG Spec, Rolling Averages).
    """
    df0 = df.copy()

    # Formatting Date & Sorting
    df0['Date'] = pd.to_datetime(df0['Date'], dayfirst=True)
    df0 = df0.sort_values(['Date']).reset_index(drop=True)

    # Conversion Rate Clipping (Removing Outliers)
    raw_rates = pd.concat([
        df0['FTHG'] / df0['HST'].replace(0, 1),
        df0['FTAG'] / df0['AST'].replace(0, 1)
    ])
    seuil_90 = raw_rates.quantile(0.90)

    df0['conv_H'] = (df0['FTHG'] / df0['HST'].replace(0, 1)).clip(0, seuil_90)
    df0['conv_A'] = (df0['FTAG'] / df0['AST'].replace(0, 1)).clip(0, seuil_90)

    # Season Context
    df0['match_number'] = df0.groupby(['season', 'HomeTeam']).cumcount() + 1
    df0['Is_start_season'] = (df0['match_number'] <= 5).astype(int)

    # Offensive Performance (Rolling 5)
    df0['HST_mean_h'] = df0.groupby(['HomeTeam'])['HST'].transform(lambda x: x.rolling(5, closed='left').mean())
    df0['AST_mean_a'] = df0.groupby(['AwayTeam'])['AST'].transform(lambda x: x.rolling(5, closed='left').mean())
    df0['conv_mean_h'] = df0.groupby(['HomeTeam'])['conv_H'].transform(lambda x: x.rolling(5, closed='left').mean())
    df0['conv_mean_a'] = df0.groupby(['AwayTeam'])['conv_A'].transform(lambda x: x.rolling(5, closed='left').mean())

    # Defensive Performance (Symmetry)
    df0['HST_allowed_mean_h'] = df0.groupby(['HomeTeam'])['AST'].transform(lambda x: x.rolling(5, closed='left').mean())
    df0['conv_allowed_mean_h'] = df0.groupby(['HomeTeam'])['conv_A'].transform(lambda x: x.rolling(5, closed='left').mean())
    df0['AST_allowed_mean_a'] = df0.groupby(['AwayTeam'])['HST'].transform(lambda x: x.rolling(5, closed='left').mean())
    df0['conv_allowed_mean_a'] = df0.groupby(['AwayTeam'])['conv_H'].transform(lambda x: x.rolling(5, closed='left').mean())

    # Corner Statistics
    df0['HC_mean_h'] = df0.groupby(['HomeTeam'])['HC'].transform(lambda x: x.rolling(5, closed='left').mean())
    df0['AC_mean_a'] = df0.groupby(['AwayTeam'])['AC'].transform(lambda x: x.rolling(5, closed='left').mean())

    # xG & xGA Specification
    df0['xG_Spec_H'] = df0['HST_mean_h'] * df0['conv_mean_h']
    df0['xG_Spec_A'] = df0['AST_mean_a'] * df0['conv_mean_a']
    df0['xGA_Spec_H'] = df0['HST_allowed_mean_h'] * df0['conv_allowed_mean_h']
    df0['xGA_Spec_A'] = df0['AST_allowed_mean_a'] * df0['conv_allowed_mean_a']

    return df0

# ==========================================
# 2. FONCTION D'ENTRAÎNEMENT
# ==========================================
def train_and_validate_model(df, features, target):
    X = df[features]
    y = df[target]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, multi_class='multinomial'))
    ])

    # Validation Croisée
    cv_scores = cross_val_score(pipeline, X, y, cv=5)

    # Entraînement final
    pipeline.fit(X, y)

    print(f"--- Model Training Report ---")
    print(f"Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"Confidence Interval: +/- {cv_scores.std() * 2:.4f}")
    print(f"Total Matches in Training: {len(df)}")
    print("-" * 30)

    return pipeline

# ==========================================
# 3. COLLECTE ET NETTOYAGE
# ==========================================
seasons = ['2021', '2122', '2223', '2324', '2425', '2526']
liste_df = []

for season in seasons:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/F1.csv"
    try:
        temp_df = pd.read_csv(url)
        temp_df['season'] = season
        liste_df.append(temp_df)
    except:
        print(f"Saison {season} non disponible.")

df_raw = pd.concat(liste_df)

# Sélection des colonnes essentielles
selected_cols = ['Date', 'season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                 'HST', 'AST', 'HC', 'AC', 'B365H', 'B365D', 'B365A']
df_clean = df_raw[selected_cols].copy()

# Traitement global
df_global = data_preparation(df_clean)
df_global['FTR_num'] = df_global['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# Définition des features finales
features = ['xG_Spec_H', 'xG_Spec_A', 'xGA_Spec_H', 'xGA_Spec_A', 'Is_start_season', 'B365H', 'B365D', 'B365A']

# Drop NaNs (premiers matchs de chaque équipe)
df_ready = df_global.dropna(subset=features + ['FTR_num'])

# ==========================================
# 4. STRATÉGIE D'ENTRAÎNEMENT (AVANT-DERNIÈRE JOURNÉE)
# ==========================================
# Trouver la date de la dernière journée disponible
last_match_date = df_ready['Date'].max()
cutoff_date = last_match_date - pd.Timedelta(days=3) # On exclut les 3 derniers jours

df_train = df_ready[df_ready['Date'] <= cutoff_date]
df_test = df_ready[df_ready['Date'] > cutoff_date]

# Entraînement du modèle
model_pipeline = train_and_validate_model(df_train, features, 'FTR_num')

# Test sur la dernière journée pour vérification
if not df_test.empty:
    y_pred = model_pipeline.predict(df_test[features])
    print(f"Accuracy sur la dernière journée : {accuracy_score(df_test['FTR_num'], y_pred):.4f}")

# =======================
# 5. SAUVEGARDE
# =======================

# 1. Le Modèle
joblib.dump(model_pipeline, 'football_model.pkl')

# 2. Les dernières statistiques pour l'App
# On crée un dictionnaire des dernières stats pour CHAQUE équipe
latest_stats = df_global.sort_values('Date').groupby('HomeTeam').tail(1)
latest_stats = latest_stats.rename(columns={'HomeTeam': 'Team'})
latest_stats = latest_stats[['Team', 'HST_mean_h', 'conv_mean_h', 'HST_allowed_mean_h', 'conv_allowed_mean_h',
                             'AST_mean_a', 'conv_mean_a', 'AST_allowed_mean_a', 'conv_allowed_mean_a', 'HC_mean_h', 'AC_mean_a']]
latest_stats.to_csv('latest_team_stats.csv', index=False)

#print("\n PRÉPARATION TERMINÉE")
#print("- Modèle sauvegardé : 'football_model.pkl'")
#print("- Stats récentes sauvegardées : 'latest_team_stats.csv'")
