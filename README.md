# Ligue 1 match predictor (Season 2025-2026)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ligue-1-predictor.streamlit.app/)

## Project description
This repository contains a professional-grade statistical forecasting engine for the French Ligue 1. The goal of this project is to apply machine learning principles to sports data to provide actionable match insights.

##  Methodology
The predictive engine utilizes a logistic regression model from the `scikit-learn` library. 
* *Classification.* Unlike linear models, this approach is optimized for categorical outcomes (Home Win, Draw, Away Win).
* *Probabilistic Output.* The model uses the `predict_proba` function to generate a nuanced probability distribution, allowing for a better assessment of match volatility.
* *Feature Set.* Data includes goals scored/conceded, team strength index, and historical performance normalized for the 18-club 2025-2026 format.

##  Tech Stack
- *Language.* Python 3.12
- *Data Science.* Pandas, Scikit-Learn, NumPy
- *Frontend.* Streamlit

##  Installation and usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/komiayi/Ligue-1.git](https://github.com/komiayi/Ligue-1.git)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the app:
   ```bash
   streamlit run app.py
