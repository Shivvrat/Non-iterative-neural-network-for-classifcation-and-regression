from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def regression_models(X_train, X_test, y_train, y_test):
    LRRegressor = LinearRegression()
    SVMRegressor = SVR()
    randomForestRegressor = RandomForestRegressor()
    dtreeRegressor = DecisionTreeRegressor()
    LRRegressor.fit(X_train, y_train)
    predicted_values = LRRegressor.predict(X_test)
    LREval = evaluate_regressor(predicted_values, y_test)
    SVMRegressor.fit(X_train, y_train)
    predicted_values = SVMRegressor.predict(X_test)
    svmEval = evaluate_regressor(predicted_values, y_test)
    randomForestRegressor.fit(X_train, y_train)
    predicted_values = randomForestRegressor.predict(X_test)
    forestEval = evaluate_regressor(predicted_values, y_test)
    dtreeRegressor.fit(X_train, y_train)
    predicted_values = dtreeRegressor.predict(X_test)
    dtreeEval = evaluate_regressor(predicted_values, y_test)
    eval = {}
    eval['LR'] = LREval
    eval['SVM'] = svmEval
    eval['Random Forest'] = forestEval
    eval['dtree'] = dtreeEval
    return eval

def evaluate_regressor(predicted_output, y_test):
    y_test = y_test.array
    rmse = metrics.mean_squared_error(y_test, predicted_output)
    mae = metrics.mean_absolute_error(y_test, predicted_output)
    return rmse, mae