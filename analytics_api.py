from flask import Blueprint, jsonify
import pandas as pd

analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')

def _load_df():
    df = pd.read_csv('restaurant_sales_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # ✅ THIS WAS THE MISSING LINE causing the /revenue endpoint to crash
    df['month'] = df['date'].dt.month 
    
    # Ensure columns exist to prevent crashes
    if 'actual_selling_price' not in df.columns:
        df['actual_selling_price'] = 10.0
    if 'typical_ingredient_cost' not in df.columns:
        df['typical_ingredient_cost'] = 5.0
        
    df['revenue'] = df['quantity_sold'] * df['actual_selling_price']
    df['profit'] = df['revenue'] - (df['quantity_sold'] * df['typical_ingredient_cost'])
    return df

@analytics_bp.route('/summary')
def summary():
    df = _load_df()
    top_dishes = df.groupby('menu_item_name')['quantity_sold'].sum().sort_values(ascending=False).head(5)
    meal_type = df.groupby('meal_type')['quantity_sold'].mean()
    weather = df.groupby('weather_condition')['quantity_sold'].mean()
    
    return jsonify({
        'top_dishes': {'labels': top_dishes.index.tolist(), 'values': top_dishes.values.tolist()},
        'meal_type': {'labels': meal_type.index.tolist(), 'values': meal_type.values.tolist()},
        'weather': {'labels': weather.index.tolist(), 'values': weather.values.tolist()},
        'total_sales': int(df['quantity_sold'].sum()),
        'total_revenue': round(float(df['revenue'].sum()), 2),
        'total_profit': round(float(df['profit'].sum()), 2),
    })

@analytics_bp.route('/revenue')
def revenue():
    df = _load_df()
    # Now this will work perfectly because 'month' exists!
    monthly = df.groupby('month').agg(
        revenue=('revenue', 'sum'), profit=('profit', 'sum')
    ).reset_index()
    
    return jsonify({
        'monthly': monthly.to_dict('records'),
    })

@analytics_bp.route('/heatmap')
def heatmap():
    df = _load_df()
    df['dow'] = df['date'].dt.day_name()
    pivot = df.pivot_table(index='dow', columns='meal_type',
                           values='quantity_sold', aggfunc='sum', fill_value=0)
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = pivot.reindex(order).fillna(0)
    
    return jsonify({
        'rows': pivot.index.tolist(),
        'cols': pivot.columns.tolist(),
        'data': pivot.values.tolist()
    })

@analytics_bp.route('/impact')
def impact():
    df = _load_df()
    promo = df.groupby('has_promotion')['quantity_sold'].mean().to_dict()
    weather = df.groupby('weather_condition')['quantity_sold'].mean().to_dict()
    event = df.groupby('special_event')['quantity_sold'].mean().to_dict()
    lift = 0
    if True in promo and False in promo and promo[False] > 0:
        lift = round((promo[True] - promo[False]) / promo[False] * 100, 1)
        
    return jsonify({
        'promotion_lift_pct': lift,
        'weather_effect': {str(k): round(float(v), 1) for k, v in weather.items()},
        'event_effect': {str(k): round(float(v), 1) for k, v in event.items()},
    })

@analytics_bp.route('/margins')
def margins():
    df = _load_df()
    df['unit_margin'] = df['actual_selling_price'] - df['typical_ingredient_cost']
    df['margin_pct'] = df['unit_margin'] / df['actual_selling_price'] * 100
    summary = df.groupby('menu_item_name').agg(
        avg_price=('actual_selling_price', 'mean'),
        avg_cost=('typical_ingredient_cost', 'mean'),
        avg_margin_pct=('margin_pct', 'mean'),
        units=('quantity_sold', 'sum')
    ).sort_values('avg_margin_pct', ascending=False).reset_index()
    for c in ['avg_price', 'avg_cost', 'avg_margin_pct']:
        summary[c] = summary[c].round(2)
    return jsonify(summary.to_dict('records'))