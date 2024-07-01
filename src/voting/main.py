#import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Reading the csv file
data = pd.read_csv('80_WO_KM_8_WC.csv')

X = data.drop(['Total wet weight bean','Clone name + Refcode'], axis=1)  # Features
y = data['Total wet weight bean']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the individual models
rf_model = RandomForestRegressor(bootstrap= False, max_depth= None, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 200)
knn_model = KNeighborsRegressor(algorithm='auto', leaf_size=20, n_neighbors=9, p=1, weights='distance')
huber_model = HuberRegressor(alpha= 0.1, epsilon= 1.35, max_iter= 500)
nn_model = MLPRegressor(
    activation='relu', 
    alpha=0.01, 
    hidden_layer_sizes=(100, 50), 
    learning_rate='invscaling', 
    solver='adam'
)

# Create the VotingRegressor --> We saw that without NN, the metrics are better, so we won't use it in the votingRegressor
voting_regressor = VotingRegressor(estimators=[
    ('rf', rf_model),
    ('knn', knn_model),
    ('huber', huber_model),
])

# Fit the ensemble model on the training data
voting_regressor.fit(X_train, y_train)

# Predict the test set
y_pred = voting_regressor.predict(X_test)

# Evaluate the performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# Calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_test, y_pred): 
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')