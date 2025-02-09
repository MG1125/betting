from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Fetch player ID using the nba_api
def get_player_id(player_name):
    player_dict = players.find_players_by_full_name(player_name)
    if player_dict:
        return player_dict[0]['id']
    return None

# Fetch player stats using the nba_api for a specific season
def fetch_player_stats(player_name, season):
    player_id = get_player_id(player_name)
    if player_id is None:
        return None
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        if gamelog.empty:
            return None
        player_stats = gamelog[['MIN', 'FGA', 'FTA', 'PTS', 'AST', 'REB', 'FG3M']].mean()
        return {
            'MIN': player_stats['MIN'],
            'FGA': player_stats['FGA'],
            'FTA': player_stats['FTA'],
            'POINTS': player_stats['PTS'],
            'ASSISTS': player_stats['AST'],
            'REBOUNDS': player_stats['REB'],
            'THREE_POINTERS': player_stats['FG3M'],
        }
    except Exception:
        return None

# Fetch historical data for the last 3 years (seasons)
def fetch_historical_data(player_name):
    seasons = ['2021-22', '2022-23', '2023-24']
    player_data = []
    for season in seasons:
        stats = fetch_player_stats(player_name, season)
        if stats:
            stats['SEASON'] = season
            player_data.append({'PLAYER_NAME': player_name, **stats})
    return pd.DataFrame(player_data) if player_data else pd.DataFrame()

# Train the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Make predictions
def make_predictions(player_name):
    historical_data = fetch_historical_data(player_name)
    if historical_data.empty:
        return None

    features = ['MIN', 'FGA', 'FTA']
    targets = {'PTS': 'POINTS', 'AST': 'ASSISTS', 'REB': 'REBOUNDS', 'FG3M': 'THREE_POINTERS'}
    X = historical_data[features]
    y = historical_data[[targets[key] for key in targets]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {}
    for target, target_col in targets.items():
        model = train_model(X_scaled, historical_data[target_col])
        models[target] = model

    latest_data = historical_data[features].iloc[-1].values.reshape(1, -1)
    latest_data_scaled = scaler.transform(latest_data)

    predictions = {}
    for stat, model in models.items():
        predictions[stat] = model.predict(latest_data_scaled)[0]
    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    if request.method == 'POST':
        players = [request.form.get(f'player{i}') for i in range(1, 4)]
        for player in players:
            if player:
                predictions[player] = make_predictions(player)
    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
