import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import joblib

# Load data
df = pd.read_csv('restaurant_sales_data.csv')

# Feature Engineering
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

if 'key_ingredients_tags' in df.columns:
    df = df.drop(['key_ingredients_tags'], axis=1)

categorical_cols = ['restaurant_type', 'menu_item_name', 'meal_type',
                    'weather_condition', 'has_promotion', 'special_event']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

drop_cols = ['quantity_sold', 'date']
if 'revenue' in df.columns:
    drop_cols.append('revenue')

X = df.drop(drop_cols, axis=1)
y = df['quantity_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train 5 Models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

print("Training Models...")
metrics_results = {} 

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    print(f"{name:20} R²={r2:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}")
    
    joblib.dump(model, f'{name.replace(" ", "_")}.pkl')
    metrics_results[name] = {'mae': float(mae), 'r2': float(r2), 'rmse': float(rmse)}

# Save the BEST model (XGBoost) as the app default
joblib.dump(models['XGBoost'], 'model.pkl')
joblib.dump(list(X.columns), 'feature_cols.pkl')

# ✅ THIS IS THE LINE YOU WERE MISSING
joblib.dump(metrics_results, 'model_metrics.pkl')

print("✅ All models trained and saved! XGBoost selected as the best model.")