import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential, Model
from Data_pre import train_data, test_data, features, targets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Define the CSV file to store the results
csv_filename = 'bho_lstm_results.csv'

# Function to create LSTM model
def create_lstm_model(input_shape):
    inputs = Input(shape=input_shape)  # Input layer with specified shape
    x = LSTM(units=50, return_sequences=True)(inputs)
    x = LSTM(units=50, return_sequences=True)(x)
    x = LSTM(units=50)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)  # Create a Model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Bee Colony Optimization Feature Selection Function
def bco_feature_selection(X_train, y_train, features):
    num_features = X_train.shape[1]
    num_selected_features = int(num_features * 0.5)  # Select top 50% features

    # Initialize bees (solutions)
    population_size = 50
    population = np.random.choice([0, 1], size=(population_size, num_features))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Fitness function using Random Forest Regressor
    def evaluate_fitness(selected_indices):
        selected_features = [features[i] for i in range(num_features) if selected_indices[i] == 1]
        if len(selected_features) == 0:
            return np.inf  # Return a high value if no features are selected

        # Select columns based on selected indices
        X_train_selected = X_train[:, selected_indices == 1]
        X_val_selected = X_val[:, selected_indices == 1]

        # Train Random Forest Regressor
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train_selected, y_train)

        # Evaluate on validation set
        y_pred = rf.predict(X_val_selected)
        mse = mean_squared_error(y_val, y_pred)
        return mse

    # BCO algorithm iterations
    num_iterations = 25
    for iteration in range(num_iterations):
        # Evaluate fitness for each bee
        fitness_values = np.array([evaluate_fitness(bee) for bee in population])

        # Select best bee (solution)
        best_bee_index = np.argmin(fitness_values)
        best_bee = population[best_bee_index]

        # Update other bees (solutions) based on best bee
        for i in range(population_size):
            if i != best_bee_index:
                # Update bee (solution) based on best bee
                crossover_point = np.random.randint(num_features)
                population[i, :crossover_point] = best_bee[:crossover_point]
                population[i, crossover_point:] = best_bee[crossover_point:]

                # Apply mutation with probability 0.1
                for j in range(num_features):
                    if np.random.rand() < 0.1:
                        population[i, j] = 1 - population[i, j]

        # Extract selected features from the best bee
        selected_indices = best_bee
        selected_features = [features[i] for i in range(num_features) if selected_indices[i] == 1]

        print(f"Iteration {iteration+1}/{num_iterations} - Best Fitness: {fitness_values[best_bee_index]:.4f}, Selected Features: {selected_features}")

    # Final selected features from the best bee
    selected_indices = best_bee
    selected_features = [features[i] for i in range(num_features) if selected_indices[i] == 1]
    print(f"Selected Features: {selected_features}")
    return selected_features


# Create a dictionary to store the LSTM results
lstm_results = {}

# Define different prediction horizons (n values)
prediction_horizons = [1, 3, 10, 30]

# Open the CSV file in write mode
with open(csv_filename, 'w', newline='') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(['Parameter', 'Prediction Horizon (n)', 'MAE', 'MSE', 'RMSE', 'MAPE'])

    # Inside the main loop
    for param in targets:
        for n in prediction_horizons:
            print(f"Processing Parameter: {param}, Prediction Horizon (n): {n} days")
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

            #print("Data preprocessing complete.")

            # Normalize the data for X_train
            scaler_X = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)

            # Normalize the data for X_test
            X_test_scaled = scaler_X.transform(X_test)

            # Normalize the target variable
            scaler_y = MinMaxScaler()
            y_train_shifted_scaled = scaler_y.fit_transform(np.array(y_train_shifted).reshape(-1, 1))

            #print("Data normalization complete.")

            # Bee Colony Optimization Feature Selection
            selected_features = bco_feature_selection(X_train_scaled, y_train_shifted_scaled.flatten(), features)
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            print("Feature selection complete.")

            # Create sequences for LSTM
            sequence_length = 30
            X_train_sequences = []
            y_train_sequences = []

            for i in range(sequence_length, len(X_train_selected)):
                X_train_sequences.append(X_train_scaled[i - sequence_length:i, :])
                y_train_sequences.append(y_train_shifted_scaled[i, 0])

            X_train_sequences = np.array(X_train_sequences)
            y_train_sequences = np.array(y_train_sequences)

            # Create and train the LSTM model
            lstm_model = create_lstm_model((X_train_sequences.shape[1], X_train_sequences.shape[2]))

            lstm_model.fit(X_train_sequences, y_train_sequences, epochs=50, batch_size=32)

            #print("LSTM model training complete.")

            # Prepare the test data for prediction
            X_test_sequences = []

            for i in range(sequence_length, len(X_test_selected)):
                X_test_sequences.append(X_test_scaled[i - sequence_length:i, :])

            X_test_sequences = np.array(X_test_sequences)
            X_test_sequences = np.reshape(X_test_sequences, (
                X_test_sequences.shape[0], X_test_sequences.shape[1], X_test_sequences.shape[2]))

            # Make predictions for the test set
            lstm_predictions_scaled = lstm_model.predict(X_test_sequences)
            lstm_predictions = scaler_y.inverse_transform(lstm_predictions_scaled)

            print("LSTM predictions made.")

            # Align the lengths of y_test and lstm_predictions
            y_test = y_test.iloc[sequence_length:].values
            lstm_predictions = lstm_predictions[:len(y_test)]  # Ensure same length

            # Calculate and display the error metrics for the current parameter and prediction horizon
            mae = mean_absolute_error(y_test, lstm_predictions)
            mse = mean_squared_error(y_test, lstm_predictions)
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
