import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error 
from sklearn.tree import DecisionTreeRegressor
import joblib

data = pd.read_csv("training_data1.csv")
x, y = data.iloc[:, :12], data.iloc[:, 13]
# print(data.iloc[:, 13])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=0)
clf = DecisionTreeRegressor(splitter="random", max_depth=24, min_samples_split=12, min_samples_leaf=6)
clf.fit(x_train, y_train)
joblib.dump(clf, 'decision_tree.model')

mse = mean_squared_error(y_test, clf.predict(x_test))
print("MSE: %4f" % mse)
