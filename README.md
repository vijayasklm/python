# python
 prediction of hours-marks 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#extracting the data
#using dataset to get the details
sample_data = pd.read_csv(r"C:\Users\VIZ\Downloads\Student Study Hour V2.csv")
sample_data.head()
#cleaning the data
sample_data.isnull()
A=sample_data.dropna(thresh=2)
A
#formatting the data
A.info()
A.dtypes
A.size
A.describe()
A.shape

#To get easy understanding on the dataset, data is plotted on the graph Here x is labelled with Hours
#and Y is labelled with Percentage of the scores
A.plot(x='Hours', y='Scores', color="r", style='+')
plt.title("Hours Vs Percentage Graph")

A.corr()

#separating the x,y values
X = A.iloc[:,:-1].values
y = A.iloc[:,1].values
X,y
#dividing the x_train,x_test,y_train,y_test using lib train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
X_train, X_test

#using linear rigreesion
regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.intercept_)
print(regr.coef_)

print("score: ",regr.score(X_test, y_test))

#linear rigreesion values
line=regr.coef_*X+regr.intercept_
line

#predicting values and plotting x,regreesion
y_pred = regr.predict(X_test)
line=regr.coef_*X+regr.intercept_
plt.scatter(X, y, color ='b')
plt.plot(X, line, color ='r')
  plt.show()
  
#calculating the error
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))
      
  #y_test,y_pred values
print(y_test,y_pred)

# You can also test with your own data
hours =0.0
a = np.array(hours).reshape(-1, 1)
own_pred = regr.predict(a)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

# You can also test with your own data
hours =9.8
a = np.array(hours).reshape(-1, 1)
own_pred = regr.predict(a)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
