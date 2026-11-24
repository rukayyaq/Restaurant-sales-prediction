import os
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

CATEGORICAL_COLS = [
    'restaurant_type', 'menu_item_name', 'meal_type',
    'weather_condition', 'has_promotion', 'special_event'
]

MODEL_REGISTRY = {
    'linear_regression': lambda: LinearRegression(),
    'decision_tree': lambda: DecisionTreeRegressor(max_depth=10, random_state=42),
    'random_forest': lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'gradient_boost': lambda: GradientBoostingRegressor(n_estimators=200, random_state=42),
    'xgboost': lambda: XGBRegressor(n_estimators=200, random_state=42)
}

def build_training_data(csv_path='restaurant_sales_data.csv'):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    exclude = {'date', 'quantity_sold', 'revenue'}
    feature_cols = [c for c in df_encoded.columns if c not in exclude]

    X = df_encoded[feature_cols]
    y = df_encoded['quantity_sold']
    return X, y, feature_cols

def train_all(csv_path='restaurant_sales_data.csv'):
    X, y, feature_cols = build_training_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    joblib.dump(feature_cols, 'feature_cols.pkl')
    results = {}
    
    print("Training models...")
    for name, factory in MODEL_REGISTRY.items():
        m = factory()
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        results[name] = {'mae': float(mae), 'r2': float(r2), 'rmse': float(rmse)}
        joblib.dump(m, f'{name}.pkl')
        print(f"{name.capitalize():<18} R2={r2:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

    # Save the best model as model.pkl
    best = max(results, key=lambda k: results[k]['r2'])
    joblib.dump(joblib.load(f'{best}.pkl'), 'model.pkl')
    
    # ✅ SAVE THE METRICS FILE (This makes the leaderboard show up!)
    joblib.dump(results, 'model_metrics.pkl')
    
    print(f"✅ All models trained and saved! {best.capitalize()} selected as the best model.")
    return results

def load_model(name='model'):
    path = 'model.pkl' if name == 'model' else f'{name}.pkl'
    if not os.path.exists(path):
        path = 'model.pkl'
    return joblib.load(path)

def predict_with_confidence(row, model):
    point = float(model.predict(row)[0])
    lower, upper = point, point
    if hasattr(model, 'estimators_'):
        try:
            preds = np.array([t.predict(row.values)[0] for t in model.estimators_])
            lower, upper = np.percentile(preds, [10, 90])
        except Exception:
            pass
    return point, float(lower), float(upper)