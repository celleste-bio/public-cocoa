

# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Initialize the Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor on the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on the testing data
predictions = rf_regressor.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Feature importance
feature_importance = rf_regressor.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)
y_test.info()

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
