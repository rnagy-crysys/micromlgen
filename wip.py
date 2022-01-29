from sklearn.datasets import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from micromlgen import port, port_testset


if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    X, y = load_iris(return_X_y=True)
    #regr = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5).fit(X, y)
    #regr = RandomForestRegressor(n_estimators=2, max_depth=10, min_samples_leaf=5).fit(X, y)
    regr = LogisticRegression(max_iter=100).fit(X, y)
    y_pred = regr.predict(X)

    with open('wip.txt', 'w') as file:
        file.write(port(regr, classname='DecisionTreeRegressor'))

    print(port_testset(X[:10], y_pred[:10], classname='BostonHousing'))