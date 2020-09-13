# ---------- Importing the libraries
import numpy
import pandas

# ---------- Import the dataset
dataset = pandas.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ---------- Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columnTransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = numpy.array(columnTransformer.fit_transform(X))

# ---------- Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size= 0.2, random_state = 0)

# ---------- Training the multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ---------- Prediction the test set results
yPredictions = regressor.predict(X_test)
numpy.set_printoptions(precision=2)
print(numpy.concatenate((yPredictions.reshape(len(yPredictions),1), y_test.reshape(len(y_test),1)), axis=1))