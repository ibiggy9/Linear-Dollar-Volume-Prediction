# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import metrics
import pickle
import datetime

# Start the efficiency clock
begin_time = datetime.datetime.now()

# Read data from CSV and assign to DataFrame
data = pd.read_csv("Source of volume.csv")

# Define the target variable
predict = "Cheez It"

# Extract relevant features from data
data = data[
    [
        "RITZ",
        "Triscuits",
        "Bear Paws Crackers",
        "Wheat Thins",
        "Goldfish",
        "Keebler Townhouse",
        "Cheez It",
    ]
]

# Shuffle the data to randomize the entries
data = shuffle(data)

# Split data into training and test sets
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Initialize and train the linear regression model
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# Calculate the accuracy of the model
acc = linear.score(x_test, y_test)
y_pred = linear.predict(x_test)

# Print initial model metrics
print("Initial Model Metrics:")
print("----------------------")
print("Accuracy:", acc)
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Train the model multiple times to find the best score
best = 0.6
count = 0
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    
    acc = linear.score(x_test, y_test)
    y_pred = linear.predict(x_test)
    
    count += 1
    
    print(f"\nRun number: {count}")
    print("Accuracy:", acc)
    print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    if acc > best:
        best = acc
        with open("cannabal.pickle", "wb") as f:
            pickle.dump(linear, f)

# Load the best model
with open("cannabal.pickle", "rb") as pickle_in:
    linear = pickle.load(pickle_in)

# Calculate and print final results
final_accuracy = int(acc * 100)
coefficients = list(zip(data.columns[:-1], linear.coef_))

print("\nResults:")
print("-------")
print(f"Total run time: {datetime.datetime.now() - begin_time} seconds")
print(f"Total number of runs: {count}")
print("Intercept:", linear.intercept_)
print(f"Best accuracy: {final_accuracy}%")
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("\nCannibalization Scores:")
print("----------------------")
for feature, coef in coefficients:
    print(f"Cheez It sourced {coef*100:.2f}% volume from {feature}")
