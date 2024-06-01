import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import warnings
import csv
from PyACO.ACOSolver import ACOSolver
from Data_pre import train_data, test_data, features, targets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set display option to show all columns
pd.set_option('display.max_columns', None)
# Create a dictionary to store the LSTM results
lstm_results = {}

# Define different prediction horizons (n values)
prediction_horizons = [1, 3, 10, 30]


# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input layer with specified shape
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Define the CSV file to store the results
csv_filename = 'aco_lstm_results.csv'


# ACO Feature Selection Function
def aco_feature_selection(X_train, y_train, features):
    # Convert the graph representation
    graph = [[1 if i in features else 0 for i in range(len(features))] for _ in range(len(X_train))]

    # Define the ACO parameters
    num_ants = 10
    num_iterations = 100

    # Create an instance of ACOSolver
    aco_solver = ACOSolver(num_of_ants=num_ants, num_of_vertexes=len(features), Q=100, alpha=2, beta=3,
                           rou=0.9, max_iterations=num_iterations, initial_vertex=0, tau_min=0.1, tau_max=1000,
                           ant_prob_random=0.1, super_not_change=30, plot_time=0.2)

    # Run the ACO algorithm
    aco_solver.run_aco()

    # Retrieve the pheromone matrix after the ACO algorithm
    pheromone_matrix = aco_solver.m_graph.m_edge_pheromone

    # Calculate the sum of pheromone values for each feature
    sum_pheromones = np.sum(pheromone_matrix, axis=0)

    # Sort features based on sum of pheromones
    sorted_indices = np.argsort(sum_pheromones)[::-1]

    # Select top features based on sorted indices
    num_selected_features = int(len(features) * 0.5)  # Select top 50% features
    selected_features = [features[idx] for idx in sorted_indices[:num_selected_features]]
    print(f"Selected Features: {selected_features}")

    return selected_features


# Open the CSV file in write mode
with open(csv_filename, 'w', newline='') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(['Parameter', 'Prediction Horizon (n)', 'MAE', 'MSE', 'RMSE', 'MAPE'])

    for param in targets:
        for n in prediction_horizons:
            # Split data into X (features) and y (target)
            X_train = train_data[features]
            y_train = train_data[param]  # Target is the current parameter
            X_test = test_data[features]
            y_test = test_data[param]  # Target is the current parameter

            # Create new target variables for each prediction horizon
            y_train_shifted = y_train.shift(-n)  # Shift target values n days into the future

            # Remove rows with NaN in the shifted target variable
            X_train = X_train[:-n]
            y_train_shifted = y_train_shifted.dropna()

            # Drop NaN values from X_train and y_train
            X_train.dropna(inplace=True)
            y_train_shifted.dropna(inplace=True)

            # Normalize the data for X_train
            scaler_X = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)

            # Normalize the data for X_test
            X_test_scaled = scaler_X.transform(X_test)

            # Normalize the target variable
            scaler_y = MinMaxScaler()
            y_train_shifted_scaled = scaler_y.fit_transform(np.array(y_train_shifted).reshape(-1, 1))

            # ACO Feature Selection
            selected_features = aco_feature_selection(X_train_scaled, y_train_shifted_scaled.flatten(), features)
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]

            # Create sequences for LSTM
            sequence_length = 30
            X_train_sequences = []
            y_train_sequences = []

            for i in range(sequence_length, len(X_train_selected)):
                X_train_sequences.append(X_train_scaled[i - sequence_length:i, :])
                y_train_sequences.append(y_train_shifted_scaled[i, 0])

            X_train_sequences = np.array(X_train_sequences)
            y_train_sequences = np.array(y_train_sequences)

            # Reshape X_train_sequences to match LSTM input shape
            X_train_sequences = np.reshape(X_train_sequences, (
                X_train_sequences.shape[0], X_train_sequences.shape[1], X_train_sequences.shape[2]))

            # Create and train the LSTM model
            lstm_model = create_lstm_model((X_train_sequences.shape[1], X_train_sequences.shape[2]))
            lstm_model.fit(X_train_sequences, y_train_sequences, epochs=50, batch_size=32, verbose=0)

            # Prepare the test data for prediction
            X_test_sequences = []

            for i in range(sequence_length, len(X_test_selected) - n):
                X_test_sequences.append(X_test_scaled[i - sequence_length:i, :])

            X_test_sequences = np.array(X_test_sequences)
            X_test_sequences = np.reshape(X_test_sequences, (
                X_test_sequences.shape[0], X_test_sequences.shape[1], X_test_sequences.shape[2]))

            # Make predictions for the test set
            lstm_predictions_scaled = lstm_model.predict(X_test_sequences)
            lstm_predictions = scaler_y.inverse_transform(lstm_predictions_scaled)

            # Align the lengths of y_test and lstm_predictions
            y_test = y_test.iloc[sequence_length:-n].values
            lstm_predictions = lstm_predictions[:len(y_test)]  # Ensure same length

            # Calculate and display the error metrics for the current parameter and prediction horizon
            nan_indices = np.isnan(y_test)
            y_test = y_test[~nan_indices].flatten()
            lstm_predictions = lstm_predictions[~nan_indices].flatten()

            mae = np.mean(np.abs(y_test - lstm_predictions))
            mse = np.mean((y_test - lstm_predictions) ** 2)
            rmse = np.sqrt(mse)

            def calculate_mape(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            mape = calculate_mape(y_test, lstm_predictions)

            # Display metrics for the current parameter and prediction horizon
            print(f"Parameter: {param}, Prediction Horizon (n): {n} days")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
            print()

            # Store results in the dictionary
            lstm_results[(param, n)] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape
            }

            # Append the results to the CSV file
            csv_writer.writerow([param, n, mae, mse, rmse, mape])

# Save the results in a CSV file
print(f"LSTM Results saved to {csv_filename}")
