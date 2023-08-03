import pandas as pd
import joblib

# Load the trained model
model = joblib.load("saved_model.pkl")

# Read the validation dataset into a pandas DataFrame
validation_data = pd.read_csv("ValidationDataset.csv", sep=";")

# Assuming you have already defined the feature_columns
feature_columns = validation_data.columns[:-1]
X_val = validation_data[feature_columns]

# Perform predictions on the validation data
predictions = model.predict(X_val)

# Print the predictions (you can do further processing or evaluation as needed)
print(predictions)
