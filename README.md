# 🏏 IPL Match Winner Predictor

A machine learning app that predicts IPL match winners based on 
**actual playing XI** — not just team names. Enter 11 players for 
each team and get win probability backed by 16 years of IPL statistics.

## 🚀 Live Demo

👉 [Open App]([#](https://ipl-playing-11-predictor.streamlit.app/)) 

## 🎯 What Makes This Unique

Most IPL predictors use just team names and toss data.
This model uses **actual player statistics** engineered from 
260,920 ball-by-ball deliveries across 1,090 matches (2008–2024):

- Individual batting strike rates and average runs per match
- Individual bowling economy rates and bowling strike rates  
- Team batting strength score (weighted average of XI)
- Team bowling strength score (weighted average of XI)
- Head-to-head win rates between teams
- Venue-specific win rates per team
- Recent form (last 5 matches)

## 📊 Dataset

| File | Rows | What it contains |
|---|---|---|
| matches.csv | 1,090 | Match results, toss, venue, winner |
| deliveries.csv | 260,920 | Ball-by-ball data for player stats |

**Source:** IPL Complete Dataset 2008–2024 (Kaggle)

## 🤖 Model

- **Algorithm:** Logistic Regression
- **Features:** 14 engineered features
- **CV Accuracy:** ~52% (realistic for cricket prediction)
- **Training Data:** 872 matches | **Test Data:** 218 matches

**Note:** 50-55% accuracy is realistic and honest for pre-match 
cricket prediction. Even professional betting models rarely exceed 
62%. Cricket has inherent randomness — pitch conditions, weather, 
player form on the day — that no model can fully capture.

## 📱 App Features

- Select teams from all 14 IPL franchises
- Enter playing XI for both teams
- Choose venue and toss details
- Get win probability for both teams
- View team batting vs bowling strength comparison
- See individual player career stats (runs, SR, wickets, economy)

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn, joblib
- **Deployment** — Streamlit Cloud
- **Data Processing** — Ball-by-ball feature engineering

## 📁 Project Structure

5.IPL_Match_Predictor/
├── data/
│ ├── matches.csv
│ └── deliveries.csv
├── models/
│ ├── ipl_model.pkl
│ ├── le_venue.pkl
│ ├── le_team.pkl
│ ├── batting_stats.csv
│ ├── bowling_stats.csv
│ ├── feature_cols.json
│ ├── teams.json
│ └── venues.json
├── notebooks/
│ └── ipl_analysis.ipynb
├── app.py
└── requirements.txt


## 💡 How to Run Locally

```bash
git clone https://github.com/vedant4687/ipl-match-predictor
cd ipl-match-predictor
pip install -r requirements.txt
streamlit run app.py
```

## 🔍 Feature Engineering Details

**Batting Strength Score per player:**

score = (strike_rate × 0.6) + (avg_runs_per_match × 0.4)


**Bowling Strength Score per player:**

score = (1/economy × 100) + (1/bowling_strike_rate × 50)


**Team Score = Average of all XI player scores**

Minimum thresholds applied:
- Batters: 100+ balls faced
- Bowlers: 120+ balls bowled
