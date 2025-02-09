from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from time import sleep
from datetime import datetime

# Function to fetch player data
def fetch_player_game_logs(season):
    """
    Fetch game logs for all players in a specific season.
    """
    players_list = players.get_active_players()  # Fetch list of all players
    player_ids = [player['id'] for player in players_list]  # Extract player IDs
    season_data = []

    for player_id in player_ids:
        try:
            # Fetch game logs for the given player and season
            game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
            season_data.append(game_log)
        except Exception as e:
            print(f"Error fetching data for player {player_id}: {e}")
            continue

    return pd.concat(season_data, ignore_index=True) if season_data else pd.DataFrame()

def fetch_season_data(start_year):
    """
    Fetch all player stats from a given start year to the current season.
    """
    all_data = []
    current_year = datetime.now().year

    for year in range(start_year, current_year + 1):
        print(f"Fetching data for the {year} season...")
        try:
            season = f"{year}-{str(year + 1)[-2:]}"  # Format like "2005-06"
            data = fetch_player_game_logs(season)
            all_data.append(data)
        except Exception as e:
            print(f"Error fetching data for the {year} season: {e}")
        sleep(1)  # Pause to avoid being rate-limited

    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

# Function to preprocess data
def preprocess_data(data):
    """
    Prepare data for model training.
    """
    data = data[['PTS', 'AST', 'REB', 'FG3M']]  # Keep only relevant columns
    data = data.dropna()  # Remove rows with missing values
    X = data[['AST', 'REB', 'FG3M']]  # Features
    y = data['PTS']  # Target (Points)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to build and evaluate the ensemble model
def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Build an ensemble model using Random Forest and Linear Regression.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_predictions = lr_model.predict(X_test_scaled)

    # Ensemble Prediction: Averaging predictions
    ensemble_predictions = (rf_predictions + lr_predictions) / 2

    # Evaluation
    mse = mean_squared_error(y_test, ensemble_predictions)
    r2 = r2_score(y_test, ensemble_predictions)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")

# Main Execution
if __name__ == "__main__":
    try:
        print("Fetching player data from 2005 to the current season...")
        raw_data = fetch_season_data(2005)
        print("Data fetching complete. Preprocessing data...")
        
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(raw_data)
        
        # Build and evaluate model
        print("Building and evaluating model...")
        build_and_evaluate_model(X_train, X_test, y_train, y_test)
        print("Model evaluation complete.")
    except Exception as e:
        print(f"An error occurred: {e}")
