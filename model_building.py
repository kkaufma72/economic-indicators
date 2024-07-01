from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_model(X_train, y_train, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor()
    elif model_type == 'xgboost':
        model = XGBRegressor()
    else:
        raise ValueError("Invalid model_type")
    model.fit(X_train, y_train)
    return model
