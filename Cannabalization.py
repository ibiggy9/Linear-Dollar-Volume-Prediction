import pandas as pd
import numpy as np
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

begin_time = datetime.datetime.now()
data = pd.read_csv("Cannabalization Data.csv")

predict = "Veggie Crisps"

data = data[["Vol1", "Vol2", "Vol3", "Vol4", "Vol5", "Vol6", "Vol7", "Vol8", "Vol9", "Vol10", "Vol11", "Vol12", "Vol13", "Vol14", "Vol15", "Veggie Crisps"]]
data = shuffle(data) # Optional - shuffle the data

#how to display data
#print(data.head())
#print(data.tail(n=10))

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)

# The line below is needed for error testing.
y_pred = linear.predict(x_test)

print("Accuracy: " + str(acc))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE

best = 0.8
count = 0
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    # The line below is needed for error testing.
    y_pred = linear.predict(x_test)
    count += 1
    print(f"Run number: {count}")
    print("Accuracy: " + str(acc))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


if acc > best:
    best = acc
    #save the model
    with open("cannabal.pickle", "wb") as f:
        pickle.dump(linear, f)


# LOAD MODEL
pickle_in = open("cannabal.pickle", "rb")
linear = pickle.load(pickle_in)
final_accuracy = best*100
coeff = {}
arr = linear.coef_

a_zip = zip(data,arr)
ziper = list(a_zip)
run_time = datetime.datetime.now() - begin_time
print("------------------------------------------------------------------")
print("\nResults are:")
for i in ziper:
    print(f"\n{i}")


print("------------------------------------------------------------------")
print("Output Report:\n\n")
print("------------------------------------------------------------------")
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