# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load your dataset
# Replace 'your_data.csv' with the path to your CSV file
data = pd.read_csv('/home/public-cocoa/data/dataForModel/80/80_O_ME.csv')

# Dropping 'Clone name' and 'Refcode' columns
data = data.drop(['Clone name + Refcode'], axis=1)

# Assuming the last column is the target variable
X = data.iloc[:, :-1]  # Features
y = data['Total wet weight bean']  # Target variable

# Convert string columns to numerical using LabelEncoder
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree regressor
dt_regressor = DecisionTreeRegressor(random_state=42)

# Train the regressor on the training data
dt_regressor.fit(X_train, y_train)

# Make predictions on the testing data
predictions = dt_regressor.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Feature importance (not available for single decision tree)
# Decision trees don't provide direct feature importance like Random Forests
# If you want feature importance, you may need to use other methods or ensemble methods like Random Forests

# Initialize the Decision Tree regressor with pruning parameters
dt_regressor_pruned = DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42)

# Train the pruned regressor on the training data
dt_regressor_pruned.fit(X_train, y_train)

# Make predictions on the testing data
predictions_pruned = dt_regressor_pruned.predict(X_test)

# Calculate evaluation metrics for the pruned tree
mse_pruned = mean_squared_error(y_test, predictions_pruned)
r2_pruned = r2_score(y_test, predictions_pruned)
print("Pruned Tree - Mean Squared Error:", mse_pruned)
print("Pruned Tree - R-squared:", r2_pruned)

