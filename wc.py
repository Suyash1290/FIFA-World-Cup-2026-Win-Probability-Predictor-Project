import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

file_path = r"D:\Round 1\World Cup predictor\international_matches.csv"
data = pd.read_csv(file_path)

X = data[['home_team_fifa_rank', 'away_team_fifa_rank', 'home_team_total_fifa_points', 'away_team_total_fifa_points']]
y = data['home_team_result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

def predict_match_outcome(home_team_rank, away_team_rank, home_team_points, away_team_points):
    input_data = scaler.transform([[home_team_rank, away_team_rank, home_team_points, away_team_points]])
    probabilities = model.predict_proba(input_data)[0]
    return probabilities

home_team_rank = int(input("Enter home team FIFA rank: "))
away_team_rank = int(input("Enter away team FIFA rank: "))
home_team_points = int(input("Enter home team total FIFA points: "))
away_team_points = int(input("Enter away team total FIFA points: "))

probabilities = predict_match_outcome(home_team_rank, away_team_rank, home_team_points, away_team_points)
print("Probability of home team winning:", probabilities[0])
print("Probability of away team winning:", probabilities[1])
print("Probability of draw:", probabilities[2])
