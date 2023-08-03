from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

# Initialize Spark session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Read the training dataset (TrainingDataset.csv) into a Spark DataFrame
training_data = spark.read.csv("TrainingDataset.csv", header=True, inferSchema=True, sep=";")

# Prepare the data for training by assembling features into a vector
feature_columns = training_data.columns[:-1]  # Exclude the last column which is the target label (quality)
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_training_data = assembler.transform(training_data)

# Convert Spark DataFrame to pandas DataFrame
import pandas as pd
pandas_training_data = assembled_training_data.toPandas()

# Separate features (X) and labels (y)
X = pandas_training_data[feature_columns]
y = pandas_training_data['quality']

# Create a Logistic Regression model and fit it to the training data
lr = LogisticRegression(max_iter=10)
model = lr.fit(X, y)

# Evaluate the model on the validation set
# Assuming you have the validation data in a pandas DataFrame as well
validation_data = pd.read_csv("ValidationDataset.csv", sep=";")
X_val = validation_data[feature_columns]
y_val = validation_data['quality']
predictions = model.predict(X_val)
f1_score_val = f1_score(y_val, predictions, average='weighted')

# Save the trained model
joblib.dump(model, "saved_model.pkl")

# Print the F1 score
print("F1 Score:", f1_score_val)

# End the Spark session
spark.stop()
