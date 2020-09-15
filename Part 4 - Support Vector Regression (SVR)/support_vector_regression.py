# Support Vector regression (SVR)

# ---------- Importing libraries
import numpy
import pandas

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
# ---------- Visualising the SVR results
# ---------- Visualising the SVR results (for higher resolution and smoother curve)