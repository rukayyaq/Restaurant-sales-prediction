import os
import pandas as pd
import joblib
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv

from models import load_model, predict_with_confidence
from analytics_api import analytics_bp

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-fallback-key')

if not os.path.exists('model.pkl') or not os.path.exists('feature_cols.pkl'):
    print("🛠️ Model not found! Training once...")
    os.system('python train_model.py')

model = load_model('model')
with open('feature_cols.pkl', 'rb') as f:
    FEATURE_COLS = joblib.load(f)

app.register_blueprint(analytics_bp)

df_analytics = pd.read_csv('restaurant_sales_data.csv')


def build_feature_row(src):
    def g(key, default=''):
        try:
            if hasattr(src, 'get'):
                v = src.get(key, default)
            else:
                v = src[key]
        except Exception:
            v = default
        return v if v is not None else default

    restaurant_type = g('restaurant_type')
    menu_item       = g('menu_item')
    meal_type       = g('meal_type')
    date_str        = g('date')
    selling_price   = float(g('selling_price', 0) or 0)
    weather         = g('weather')
    promotion       = g('promotion')
    special_event   = g('event')

    dt = pd.to_datetime(date_str)
    day_of_week = dt.dayofweek
    month = dt.month
    day = dt.day
    is_weekend = int(day_of_week >= 5)

    row = pd.DataFrame(0, index=[0], columns=FEATURE_COLS)

    cat_mapping = {
        'restaurant_type': restaurant_type,
        'menu_item_name': menu_item,
        'meal_type': meal_type,
        'weather_condition': weather,
        'has_promotion': promotion,
        'special_event': special_event
    }
    for prefix, value in cat_mapping.items():
        col_name = f"{prefix}_{value}"
        if col_name in FEATURE_COLS:
            row[col_name] = 1

    row['typical_ingredient_cost'] = 0
    row['observed_market_price'] = selling_price
    row['actual_selling_price'] = selling_price
    row['day_of_week'] = day_of_week
    row['month'] = month
    row['day'] = day
    row['is_weekend'] = is_weekend

    return row[FEATURE_COLS]

def _load_metrics():
    try:
        return joblib.load('model_metrics.pkl')
    except Exception:
        return {}

@app.route('/')
def home():
    return render_template('index.html', metrics=_load_metrics())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        row = build_feature_row(request.form)
        point, low, high = predict_with_confidence(row, model)
        prediction = int(round(point))

        alert_message = None
        if prediction > 300:
            alert_message = "⚠️ High volume predicted! Ensure staff and ingredients are prepared."
        elif prediction < 50:
            alert_message = "📉 Low volume predicted. Consider running a promotion or reducing prep."

        return render_template('index.html',
                               prediction=prediction,
                               alert=alert_message,
                               confidence_low=int(round(low)),
                               confidence_high=int(round(high)),
                               metrics=_load_metrics(),
                               input_data=request.form)
    except Exception as e:
        return render_template('index.html',
                               error=str(e),
                               metrics=_load_metrics(),
                               input_data=request.form)

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

@app.route('/api/analytics')
def api_analytics_legacy():
    top_dishes = df_analytics.groupby('menu_item_name')['quantity_sold'].sum().sort_values(ascending=False).head(5)
    meal_type = df_analytics.groupby('meal_type')['quantity_sold'].mean()
    weather = df_analytics.groupby('weather_condition')['quantity_sold'].mean()
    return jsonify({
        'top_dishes': {'labels': top_dishes.index.tolist(), 'values': top_dishes.values.tolist()},
        'meal_type': {'labels': meal_type.index.tolist(), 'values': meal_type.values.tolist()},
        'weather': {'labels': weather.index.tolist(), 'values': weather.values.tolist()},
        'total_sales': int(df_analytics['quantity_sold'].sum())
    })

@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'POST':
        try:
            f = request.files['csv']
            if not f.filename:
                return render_template('batch_predict.html', error='No file uploaded')
            
            df = pd.read_csv(f)
            df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
            
            defaults = {
                'restaurant_type': 'Casual Dining',
                'menu_item': 'Nasi Lemak',
                'meal_type': 'Lunch',
                'date': '2025-11-24',
                'selling_price': 10.0,
                'weather': 'Sunny',
                'promotion': 'False',
                'event': 'False'
            }
            
            for col, default_val in defaults.items():
                if col not in df.columns:
                    df[col] = default_val

            preds = []
            for _, r in df.iterrows():
                row = build_feature_row(r)
                preds.append(int(round(float(model.predict(row)[0]))))
                
            df['predicted_quantity'] = preds
            out = 'static/batch_output.csv'
            df.to_csv(out, index=False)
            
            return render_template('batch_predict.html',
                                   success=True,
                                   preview=df.head(20).to_html(classes='table table-sm table-striped', index=False),
                                   download_url='/static/batch_output.csv')
        except Exception as e:
            return render_template('batch_predict.html', error=str(e))
            
    return render_template('batch_predict.html')

@app.route('/batch/sample')
def batch_sample():
    sample = pd.DataFrame([{
        'restaurant_type': 'Casual Dining', 'menu_item': 'Nasi Lemak',
        'meal_type': 'Lunch', 'date': '2025-11-24', 'selling_price': 12.5,
        'weather': 'Sunny', 'promotion': 'True', 'event': 'False',
    }])
    buf = BytesIO(sample.to_csv(index=False).encode())
    return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='sample_batch.csv')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json() or {}
        row = build_feature_row(data)
        point, low, high = predict_with_confidence(row, model)
        return jsonify({
            'prediction': int(round(point)),
            'confidence_low': int(round(low)),
            'confidence_high': int(round(high)),
            'model': type(model).__name__,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/models')
def api_models():
    return jsonify(_load_metrics())

@app.route('/api/feature_importance')
def feature_importance():
    try:
        rf = joblib.load('random_forest.pkl')
        importances = rf.feature_importances_
        
        feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)[:6]
        
        cleaned = []
        for f, val in feat_imp:
            name = f.replace('menu_item_name_', '').replace('restaurant_type_', '').replace('weather_condition_', '').replace('_', ' ')
            cleaned.append({'feature': name.title(), 'importance': round(float(val) * 100, 1)})
            
        return jsonify(cleaned)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)