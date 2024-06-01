import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense,  Input
import csv
from Data_pre import train_data, test_data, features, targets
# Define the CSV file to store the results
csv_filename = 'ga_lstm_results.csv'

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Genetic Algorithm Feature Selection Function
def ga_feature_selection(X_train, y_train, features, num_generations=50, population_size=50, mutation_rate=0.1):
    num_features = len(features)
    num_selected_features = int(num_features * 0.5)  # Select top 50% features

    # Create a mapping from feature names to their column indices
    feature_index_map = {feature: index for index, feature in enumerate(features)}

    best_individual = None
    best_fitness = float('inf')

    for _ in range(num_generations):
        population = np.random.choice([0, 1], size=(population_size, num_features))

        for i in range(population_size):
            selected_indices = np.where(population[i] == 1)[0]
            selected_features = [features[idx] for idx in selected_indices]

            # Convert feature names to column indices
            selected_indices = [feature_index_map[feature] for feature in selected_features]

            # Evaluate fitness (e.g., using mean squared error)
            X_train_selected = X_train[:, selected_indices]  # Select columns corresponding to selected features
            lstm_model = create_lstm_model((X_train_selected.shape[1], 1))  # Fix input shape
            # Train the LSTM model and calculate fitness
            # (You need to implement training and evaluation of the model here)
            fitness = evaluate_fitness(X_train_selected, y_train)

            # Update best individual and fitness
            if fitness < best_fitness:
                best_individual = population[i]
                best_fitness = fitness


        # Crossover
        for i in range(0, population_size, 2):
            parent1 = population[i]
            parent2 = population[i+1]
            crossover_point = np.random.randint(num_features)
            population[i, crossover_point:] = parent2[crossover_point:]
            population[i+1, crossover_point:] = parent1[crossover_point:]

        # Mutation
        for i in range(population_size):
            for j in range(num_features):
                if np.random.rand() < mutation_rate:
                    population[i, j] = 1 - population[i, j]

        # Get selected features from the best individual
    selected_indices = np.where(best_individual == 1)[0]
    selected_features = [features[idx] for idx in selected_indices]
    print(f"Selected Features: {selected_features}")
    return selected_features


# Dummy function for evaluating fitness (replace with actual evaluation)
def evaluate_fitness(X_train_selected, y_train):
    # Generate random fitness score for demonstration
    fitness = np.random.rand()
    return fitness


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

            # Normalize the data for X_train
            scaler_X = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)

            # Normalize the data for X_test
            X_test_scaled = scaler_X.transform(X_test)

            # Normalize the target variable
            scaler_y = MinMaxScaler()
            y_train_shifted_scaled = scaler_y.fit_transform(np.array(y_train_shifted).reshape(-1, 1))

            # Genetic Algorithm Feature Selection
            selected_features = ga_feature_selection(X_train_scaled, y_train_shifted_scaled.flatten(), features)
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

            # Create and train the LSTM model
            lstm_model = create_lstm_model((X_train_sequences.shape[1], X_train_sequences.shape[2]))

            lstm_model.fit(X_train_sequences, y_train_sequences, epochs=50, batch_size=32)

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