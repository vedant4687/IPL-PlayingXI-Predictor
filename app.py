import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ── Load Models & Data ──
model = joblib.load('models/ipl_model.pkl')
le_venue = joblib.load('models/le_venue.pkl')
le_team = joblib.load('models/le_team.pkl')
batting_stats = pd.read_csv('models/batting_stats.csv')
bowling_stats = pd.read_csv('models/bowling_stats.csv')

with open('models/feature_cols.json') as f:
    feature_cols = json.load(f)
with open('models/teams.json') as f:
    teams = json.load(f)
with open('models/venues.json') as f:
    venues = json.load(f)

# ── Helper Functions ──
def get_team_batting_strength(players):
    scores = []
    for player in players:
        if player.strip() in batting_stats['batter'].values:
            row = batting_stats[batting_stats['batter'] == player.strip()].iloc[0]
            score = (row['strike_rate'] * 0.6) + (row['avg_runs_per_match'] * 0.4)
            scores.append(score)
    return round(np.mean(scores), 2) if scores else 100.0

def get_team_bowling_strength(players):
    scores = []
    for player in players:
        if player.strip() in bowling_stats['bowler'].values:
            row = bowling_stats[bowling_stats['bowler'] == player.strip()].iloc[0]
            score = (1 / row['economy']) * 100
            if not np.isnan(row['bowling_sr']):
                score += (1 / row['bowling_sr']) * 50
            scores.append(score)
    return round(np.mean(scores), 2) if scores else 10.0

def get_player_stats(player):
    bat = batting_stats[batting_stats['batter'] == player.strip()]
    bowl = bowling_stats[bowling_stats['bowler'] == player.strip()]
    stats = {}
    if not bat.empty:
        stats['runs'] = int(bat.iloc[0]['total_runs'])
        stats['strike_rate'] = float(bat.iloc[0]['strike_rate'])
        stats['innings'] = int(bat.iloc[0]['innings'])
    if not bowl.empty:
        stats['wickets'] = int(bowl.iloc[0]['wickets'])
        stats['economy'] = float(bowl.iloc[0]['economy'])
    return stats

# ── Page Config ──
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL Match Winner Predictor")
st.markdown("Enter both playing XIs to predict the match winner based on player statistics.")
st.divider()

# ── Team Selection ──
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    team1 = st.selectbox("🔵 Team 1", teams, index=teams.index('Mumbai Indians'))

with col2:
    st.markdown("<h3 style='text-align:center; margin-top:25px'>VS</h3>", unsafe_allow_html=True)

with col3:
    team2 = st.selectbox("🔴 Team 2", teams, index=teams.index('Chennai Super Kings'))

# ── Match Details ──
col4, col5, col6 = st.columns(3)
with col4:
    venue = st.selectbox("🏟️ Venue", venues)
with col5:
    toss_winner = st.selectbox("🪙 Toss Winner", [team1, team2])
with col6:
    toss_decision = st.selectbox("🏏 Toss Decision", ["bat", "field"])

st.divider()

# ── Playing XI Input ──
col7, col8 = st.columns(2)

with col7:
    st.subheader(f"🔵 {team1} Playing XI")
    team1_players = []
    for i in range(1, 12):
        player = st.text_input(f"Player {i}", key=f"t1_p{i}", placeholder=f"Enter player {i} name")
        team1_players.append(player)

with col8:
    st.subheader(f"🔴 {team2} Playing XI")
    team2_players = []
    for i in range(1, 12):
        player = st.text_input(f"Player {i}", key=f"t2_p{i}", placeholder=f"Enter player {i} name")
        team2_players.append(player)

st.divider()

# ── Predict Button ──
if st.button("🏆 Predict Winner", type="primary", use_container_width=True):

    # Filter empty inputs
    team1_players = [p for p in team1_players if p.strip()]
    team2_players = [p for p in team2_players if p.strip()]

    if len(team1_players) < 5 or len(team2_players) < 5:
        st.error("Please enter at least 5 players for each team!")
    else:
        # Calculate strengths
        t1_bat = get_team_batting_strength(team1_players)
        t1_bowl = get_team_bowling_strength(team1_players)
        t2_bat = get_team_batting_strength(team2_players)
        t2_bowl = get_team_bowling_strength(team2_players)

        toss_team1 = 1 if toss_winner == team1 else 0
        bat_first = 1 if toss_decision == 'bat' else 0

        # Encode
        try:
            venue_enc = le_venue.transform([venue])[0]
        except:
            venue_enc = 0
        try:
            team1_enc = le_team.transform([team1])[0]
            team2_enc = le_team.transform([team2])[0]
        except:
            team1_enc = 0
            team2_enc = 1

        # Build feature vector
        features = pd.DataFrame([[
            team1_enc, team2_enc, venue_enc,
            t1_bat, t1_bowl, t2_bat, t2_bowl,
            0.5, 0.5, 0.5, 0.5, 0.5,
            toss_team1, bat_first
        ]], columns=feature_cols)

        # Predict
        prob = model.predict_proba(features)[0]
        team1_prob = round(prob[1] * 100, 1)
        team2_prob = round(prob[0] * 100, 1)

        # ── Results ──
        st.divider()
        st.subheader("🏆 Prediction Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(f"🔵 {team1}", f"{team1_prob}%", 
                     delta="Predicted Winner" if team1_prob > team2_prob else "")
        with res_col2:
            winner = team1 if team1_prob > team2_prob else team2
            st.markdown(f"<h2 style='text-align:center'>🏆<br>{winner}</h2>", unsafe_allow_html=True)
        with res_col3:
            st.metric(f"🔴 {team2}", f"{team2_prob}%",
                     delta="Predicted Winner" if team2_prob > team1_prob else "")

        # Win probability bar
        st.divider()
        st.markdown("**Win Probability**")
        st.progress(team1_prob / 100)
        st.caption(f"{team1}: {team1_prob}% | {team2}: {team2_prob}%")

        # ── Team Comparison ──
        st.divider()
        st.subheader("📊 Team Strength Comparison")

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown(f"**🔵 {team1}**")
            st.metric("Batting Strength", t1_bat)
            st.metric("Bowling Strength", t1_bowl)
            better_bat = "✅" if t1_bat > t2_bat else "❌"
            better_bowl = "✅" if t1_bowl > t2_bowl else "❌"
            st.markdown(f"Batting Edge: {better_bat} | Bowling Edge: {better_bowl}")

        with comp_col2:
            st.markdown(f"**🔴 {team2}**")
            st.metric("Batting Strength", t2_bat)
            st.metric("Bowling Strength", t2_bowl)
            better_bat = "✅" if t2_bat > t1_bat else "❌"
            better_bowl = "✅" if t2_bowl > t1_bowl else "❌"
            st.markdown(f"Batting Edge: {better_bat} | Bowling Edge: {better_bowl}")

        # ── Player Stats ──
        st.divider()
        st.subheader("👤 Player Statistics")

        p_col1, p_col2 = st.columns(2)

        with p_col1:
            st.markdown(f"**🔵 {team1} Players**")
            for player in team1_players:
                stats = get_player_stats(player)
                if stats:
                    info = []
                    if 'runs' in stats:
                        info.append(f"Runs: {stats['runs']} | SR: {stats['strike_rate']}")
                    if 'wickets' in stats:
                        info.append(f"Wkts: {stats['wickets']} | Eco: {stats['economy']}")
                    st.markdown(f"**{player}** — {' | '.join(info)}")
                else:
                    st.markdown(f"**{player}** — No IPL stats found")

        with p_col2:
            st.markdown(f"**🔴 {team2} Players**")
            for player in team2_players:
                stats = get_player_stats(player)
                if stats:
                    info = []
                    if 'runs' in stats:
                        info.append(f"Runs: {stats['runs']} | SR: {stats['strike_rate']}")
                    if 'wickets' in stats:
                        info.append(f"Wkts: {stats['wickets']} | Eco: {stats['economy']}")
                    st.markdown(f"**{player}** — {' | '.join(info)}")
                else:
                    st.markdown(f"**{player}** — No IPL stats found")