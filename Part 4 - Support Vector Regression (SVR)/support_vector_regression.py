# Support Vector regression (SVR)

# ---------- Importing libraries
import numpy
import pandas
import matplotlib.pyplot as mPlot

# ---------- Importing the dataset
dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y), 1)

# ---------- Feature scaling
from sklearn.preprocessing import StandardScaler
xScaler = StandardScaler()
yScaler = StandardScaler()
X = xScaler.fit_transform(X)
y = yScaler.fit_transform(y)

# ---------- Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf') # https://data-flair.training/blogs/svm-kernel-functions
regressor.fit(X, y);

# ---------- Predicting a new result
scaledPredictedSalary = regressor.predict(xScaler.transform([[6.5]]))
predictedSalary = yScaler.inverse_transform(scaledPredictedSalary)

# ---------- Visualising the SVR results
xITransformed = xScaler.inverse_transform(X)
yITransformed = yScaler.inverse_transform(y) 
mPlot.scatter(xITransformed, yITransformed, color='red')
mPlot.plot(xITransformed, yScaler.inverse_transform(regressor.predict(X)), color='blue')
mPlot.title('Truth or Bluff (Support Vector Regression)')
mPlot.xlabel('Position level')
mPlot.ylabel('Salary')
mPlot.show()

# ---------- Visualising the SVR results (for higher resolution and smoother curve)
xGrid = numpy.arrange(min(xITransformed), min(xITransformed), 0.1)
xGrid = xGrid.reshape(len(xGrid), 1)
mPlot.scatter(xITransformed, yITransformed, color='red')
mPlot.plot(xGrid, yScaler.inverse_transform(regressor.predict(xScaler.transform(xGrid))), color='blue')
mPlot.title('Truth or Bluff (Support Vector Regression)')
mPlot.xlabel('Position level')
mPlot.ylabel('Salary')
mPlot.show()