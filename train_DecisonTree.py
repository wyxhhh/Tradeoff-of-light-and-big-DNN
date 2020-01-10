from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import tree
from imblearn.under_sampling import RandomUnderSampler

import pydotplus
import joblib
import os

from collections import Counter

data = pd.read_csv("training_data.csv")
x, y = data.iloc[:470000, :14], data.iloc[:470000, 14]

rus = RandomUnderSampler(sampling_strategy={1: 3*Counter(y)[-1]})
x1, y1 = rus.fit_sample(x, y)
print(Counter(y1))

os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/Graphviz2.38/bin/"

x2, y2 = x1.iloc[:, :13], x1.iloc[:, 13]
y2 = y2*100

x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2, random_state=1)

# search for the best parameters
# mi = 1000
# result = []
# for i in range(12, 22):
#     for j in range(10, 16):
#         for k in range(5, 9):
#             clf = DecisionTreeRegressor(max_depth=i, min_samples_split=j, min_samples_leaf=k)
#             clf.fit(x_train, y_train)

#             mse = mean_squared_error(y_test, clf.predict(x_test))
#             print(i, j, k, "MSE: %4f" % mse)
#             if mse < mi:
#                 mi = mse
#                 result = [i, j, k]
# print(mi, result)

clf = DecisionTreeRegressor(max_depth=17, min_samples_split=15, min_samples_leaf=6)
clf.fit(x_train, y_train)

mse = mean_squared_error(y_test, clf.predict(x_test))
print("MSE: %4f" % mse)
dot = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot)
graph.write_pdf("tree1.pdf")
