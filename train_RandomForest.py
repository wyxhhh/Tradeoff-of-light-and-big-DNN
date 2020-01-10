import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import joblib
import time
from collections import Counter

from sklearn.model_selection import train_test_split 
data = pd.read_csv("training_data.csv")
x2, y2 = data.iloc[:470000, :13], data.iloc[:470000, 13]

# down-sampling step, which is not so helpful
# rus = RandomUnderSampler(sampling_strategy={1: 3*Counter(y)[-1]})
# x1, y1 = rus.fit_sample(x, y)
# # print(Counter(y1))

# x2, y2 = x1.iloc[:, :13], x1.iloc[:, 13]
y2 = y2*100

x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2, random_state=1)

# an example of training random_forest 
forest_para = {'max_depth':list(range(15, 26)), 'oob_score':[True, False], 'min_samples_leaf':list(range(3,10))}

clf = GridSearchCV(RandomForestRegressor(), param_grid=forest_para)
clf.fit(x_train, y_train)

clf = clf.best_estimator_
joblib.dump(clf, 'random_forest.model')

mse = mean_squared_error(y_test, clf.predict(x_test))
print("MSE: %4f" % mse)
# print(time.time()-t)
