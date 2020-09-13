# Data preprocessing

# ---------- Importing the libraries
import pandas as pd

# ---------- Import the dataset
dataset = pd.read_csv('../../Downloaded Data/Part1/Python/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# ---------- Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size= 0.2, random_state = 0)

# ---------- Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""