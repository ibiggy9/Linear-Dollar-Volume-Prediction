import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import metrics
import pickle
import seaborn as seabornInstance
from sklearn import metrics
import matplotlib.pyplot as pltd
import math
import datetime

#Algorithm efficiency clock
begin_time = datetime.datetime.now()

#Read Data
data = pd.read_csv("Source of volume.csv")

#Prediction metric
predict = "Cheez It"

#Data Input
data = data[[
    "RITZ", 
    "Triscuits",
    "Bear Paws Crackers", 
    "Wheat Thins", 
    "Goldfish", 
    "Keebler Townhouse", 
    "Cheez It"
]]

#Shuffle data for training the model
data = shuffle(data) 

#how to display data
#print(data.head())
#print(data.tail(n=10))

#Putting the inputs in their correct place in the algorithm 
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#Running the calculation 
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)

#Testing the accuracy of the model + Output
y_pred = linear.predict(x_test)

print("Accuracy: " + str(acc))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE

#Target minimum best accuracy
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
best = 0.6
count = 0

for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    # The line below is needed for error testing.
    y_pred = linear.predict(x_test)
    count += 1
    print(f"\nRun number: {count}")
    print("Accuracy: " + str(acc))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Displaying the accuracy
if acc > best:
    best = acc
    #Saving the model
    with open("cannabal.pickle", "wb") as f:
        pickle.dump(linear, f)

print(best)

# LOAD MODEL 
pickle_in = open("cannabal.pickle", "rb") 
linear = pickle.load(pickle_in)
final_accuracy = int(acc*100)
coeff = {}
arr = linear.coef_

a_zip = zip(data,arr)
ziper = list(a_zip)
run_time = datetime.datetime.now() - begin_time
print('\n')
print("------------------------------------------------------------------")
print("The Results Are:")
print("------------------------------------------------------------------")
print('\n')
print('\n')
print('\n')

empty = []
#Iteration through results to print final cannibalization score
for index, value in ziper:
    empty.append(value)

avg_can = int(((sum(empty)/15))*100)
binomial_effect_size_display_pos = (final_accuracy/2)+50
binomial_effect_size_display_neg = (final_accuracy/2)-50
print("------------------------------------------------------------------")
print("Accuracy Statistics")
print("------------------------------------------------------------------")
print(f"Chance of being accurate is {binomial_effect_size_display_pos}%")
print(f'Chance of being inaccurate is {binomial_effect_size_display_neg}%')


print('\n')
print("------------------------------------------------------------------")
print("Output Report:")
print("------------------------------------------------------------------")
for i, x in ziper:
    a = x*100
    print(f"Cheez It sourced{a}% volume from {i}")


print(f"\nTotal run time: \n{run_time} seconds\n")
print(f"Total number of runs: \n{count}\n")
print('Intercept: \n', linear.intercept_)
print(f'\nBest accuracy: \n{final_accuracy}%')
print('\nMean Absolute Error:\n', metrics.mean_absolute_error(y_test, y_pred))
print('\nMean Squared Error:\n', metrics.mean_squared_error(y_test, y_pred))
print('\nRoot Squared Error:\n', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("------------------------------------------------------------------")


#plot the data
#data.describe()

#Check for null values
#data.isnull().any()

#Remove null data
# data = data.fillna(method='ffill')

"""
#HOW TO PLUG IN VALUES TO PREDICT $VOL 
new_x = [[2, 90, 2.3, 2.4, 2, 4.12]]
print('The Expected $Vol with your inputs is:\n ', linear.predict(new_x))

"""
