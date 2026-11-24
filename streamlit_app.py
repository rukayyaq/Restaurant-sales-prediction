import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import datetime
import io
from fpdf import FPDF

# --- CONFIGURATION ---
st.set_page_config(page_title="Restaurant Sales Predictor", page_icon="🍽️", layout="wide")

# Custom CSS to make it look dark and professional
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    h1, h2, h3 { color: #f8fafc !important; }
    .stButton>button { background-color: #6366f1; color: white; border-radius: 8px; font-weight: bold; }
    .stButton>button:hover { background-color: #4f46e5; }
    .prediction-box { background: linear-gradient(135deg, #1e293b, #0f172a); border-left: 5px solid #10b981; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;}
    .prediction-number { font-size: 80px; color: #10b981; font-weight: bold; margin: 0; }
    .alert-box { padding: 10px; border-radius: 5px; margin-top: 10px; font-weight: bold; }
    .simulator-box { background: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL & DATA ---
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = joblib.load(f)
    df = pd.read_csv('restaurant_sales_data.csv')
    return model, feature_cols, df

try:
    model, FEATURE_COLS, df_analytics = load_assets()
except Exception as e:
    st.error(f"Error loading models. Make sure model.pkl and feature_cols.pkl exist. Error: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---
def build_feature_row(restaurant_type, menu_item, meal_type, date_str, selling_price, weather, promotion, special_event):
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

def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 10, txt="SalesPredict - AI Forecast", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)

    # Prediction Box
    pdf.set_fill_color(16, 185, 129) # Green
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 20, txt=f"PREDICTED QUANTITY: {data['prediction']} units", ln=True, align='C', fill=True)
    pdf.ln(15)

    # Details
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.cell(100, 10, txt=f"Menu Item: {data['menu_item']}", ln=False)
    pdf.cell(100, 10, txt=f"Selling Price: RM {data['price']}", ln=True)
    pdf.cell(100, 10, txt=f"Restaurant: {data['restaurant']}", ln=False)
    pdf.cell(100, 10, txt=f"Weather: {data['weather']}", ln=True)
    pdf.cell(100, 10, txt=f"Meal Type: {data['meal']}", ln=False)
    pdf.cell(100, 10, txt=f"Promotion: {data['promo']}", ln=True)
    pdf.cell(100, 10, txt=f"Date: {data['date']}", ln=False)
    pdf.cell(100, 10, txt=f"Special Event: {data['event']}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt=f"Confidence Range: {data['conf_low']} - {data['conf_high']} units", ln=True, align='C')

    return pdf.output(dest='S').encode('latin-1')

# --- MAIN APP LAYOUT ---
st.title("🍽️ Restaurant Sales Predictor")
st.markdown("Enter the details below to forecast sales using our ML models.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict Sales", "📊 Analytics Dashboard", "📂 Batch Predict"])

# ================= TAB 1: PREDICTION =================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.form("prediction_form"):
            st.subheader("Input Details")
            restaurant_type = st.selectbox("Restaurant Type", ['Casual Dining', 'Fine Dining', 'Food Stall', 'Kopitiam'])
            menu_item = st.selectbox("Menu Item", ['Nasi Lemak', 'Laksa', 'Chicken Rice', 'Char Kway Teow', 'Roti Canai', 'Teh Tarik', 'Iced Lemon Tea', 'Kaya Toast Set', 'Cendol', 'Chicken Chop', 'Mushroom Soup', 'Spaghetti Carbonara', 'Tandoori Chicken'])
            meal_type = st.selectbox("Meal Type", ['Lunch', 'Dinner'])
            date_str = st.date_input("Date", datetime.date.today())
            selling_price = st.number_input("Selling Price (RM)", min_value=1.0, value=12.50, step=0.50)
            weather = st.selectbox("Weather", ['Sunny', 'Rainy'])
            promotion = st.selectbox("Has Promotion?", ['True', 'False'])
            special_event = st.selectbox("Special Event?", ['True', 'False'])
            
            submit_button = st.form_submit_button(label="🔮 Generate Prediction")

    with col2:
        if submit_button:
            # Build features and predict
            row = build_feature_row(restaurant_type, menu_item, meal_type, date_str, selling_price, weather, promotion, special_event)
            prediction = int(round(float(model.predict(row)[0])))
            
            # Calculate confidence
            low, high = prediction, prediction
            if hasattr(model, 'estimators_'):
                preds = np.array([t.predict(row.values)[0] for t in model.estimators_])
                low, high = np.percentile(preds, [10, 90])

            # Display Result
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin:0;">Predicted Quantity</h3>
                <h1 class="prediction-number">{prediction}</h1>
                <p style="font-size:18px; color:#94a3b8;">units expected to be sold</p>
                <p style="color:#94a3b8;">Confidence Range: <b>{int(round(low))} - {int(round(high))} units</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Alerts
            if prediction > 300:
                st.warning("⚠️ High volume predicted! Ensure staff and ingredients are prepared.")
            elif prediction < 50:
                st.info("📉 Low volume predicted. Consider running a promotion.")

            # PDF Generation
            pdf_data = {
                'prediction': prediction, 'menu_item': menu_item, 'price': selling_price,
                'restaurant': restaurant_type, 'weather': weather, 'meal': meal_type,
                'promo': promotion, 'date': date_str, 'event': special_event,
                'conf_low': int(round(low)), 'conf_high': int(round(high))
            }
            pdf_bytes = generate_pdf(pdf_data)
            
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name="Sales_Prediction_Report.pdf",
                mime="application/pdf"
            )

    # --- LIVE SIMULATOR SECTION ---
    st.markdown("---")
    st.markdown("### 🎛️ Live Price Simulator")
    st.markdown("Adjust the selling price below to see how it impacts predicted sales instantly.")
    
    sim_box = st.container()
    with sim_box:
        # Using a slider outside the form so it triggers updates dynamically
        sim_price = st.slider("Simulated Selling Price (RM)", min_value=5.0, max_value=50.0, value=12.50, step=0.50)
        
        # Create a placeholder for the result so it updates instantly
        sim_result_placeholder = st.empty()
        
        # Build features using the SIMULATED price
        sim_row = build_feature_row(restaurant_type, menu_item, meal_type, date_str, sim_price, weather, promotion, special_event)
        sim_pred = int(round(float(model.predict(sim_row)[0])))
        
        # Display the result in a glowing blue box
        sim_result_placeholder.markdown(f"""
        <div style="background-color: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid rgba(56, 189, 248, 0.3);">
            <span style="font-size: 16px;">Predicted Sales: </span>
            <strong style="font-size: 28px;">{sim_pred}</strong>
            <span style="font-size: 16px;"> units</span>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 2: ANALYTICS =================
with tab2:
    st.subheader("📊 Analytics Dashboard")
    
    # Top Dishes
    col_a, col_b = st.columns(2)
    with col_a:
        top_dishes = df_analytics.groupby('menu_item_name')['quantity_sold'].sum().sort_values(ascending=False).head(5).reset_index()
        fig1 = px.bar(top_dishes, x='menu_item_name', y='quantity_sold', title="🏆 Top 5 Dishes", template="plotly_dark", color_discrete_sequence=['#6366f1'])
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_b:
        meal_type = df_analytics.groupby('meal_type')['quantity_sold'].mean().reset_index()
        fig2 = px.pie(meal_type, names='meal_type', values='quantity_sold', title="🍽️ Sales by Meal Type", template="plotly_dark", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    # Weather & Revenue
    col_c, col_d = st.columns(2)
    with col_c:
        weather = df_analytics.groupby('weather_condition')['quantity_sold'].mean().reset_index()
        fig3 = px.bar(weather, x='weather_condition', y='quantity_sold', title="🌤️ Avg Sales by Weather", template="plotly_dark", color_discrete_sequence=['#10b981'])
        st.plotly_chart(fig3, use_container_width=True)
        
    with col_d:
        df_analytics['date'] = pd.to_datetime(df_analytics['date'])
        df_analytics['month'] = df_analytics['date'].dt.month
        df_analytics['revenue'] = df_analytics['quantity_sold'] * df_analytics['actual_selling_price']
        monthly = df_analytics.groupby('month')['revenue'].sum().reset_index()
        fig4 = px.line(monthly, x='month', y='revenue', title="💹 Monthly Revenue", template="plotly_dark", markers=True)
        st.plotly_chart(fig4, use_container_width=True)

# ================= TAB 3: BATCH PREDICT =================
with tab3:
    st.subheader("📂 Batch Prediction")
    st.markdown("Upload a CSV with columns: `restaurant_type, menu_item, meal_type, date, selling_price, weather, promotion, event`")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        df_upload.columns = [c.lower().strip().replace(' ', '_') for c in df_upload.columns]
        
        # Fill missing
        defaults = {'restaurant_type': 'Casual Dining', 'menu_item': 'Nasi Lemak', 'meal_type': 'Lunch', 'date': '2025-11-24', 'selling_price': 10.0, 'weather': 'Sunny', 'promotion': 'False', 'event': 'False'}
        for col, default_val in defaults.items():
            if col not in df_upload.columns:
                df_upload[col] = default_val

        if st.button("Run Batch Prediction"):
            preds = []
            for _, r in df_upload.iterrows():
                row = build_feature_row(r['restaurant_type'], r['menu_item'], r['meal_type'], r['date'], r['selling_price'], r['weather'], r['promotion'], r['event'])
                preds.append(int(round(float(model.predict(row)[0]))))
            
            df_upload['predicted_quantity'] = preds
            st.success("✅ Predictions generated!")
            st.dataframe(df_upload)
            
            csv = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results CSV", csv, "batch_predictions.csv", "text/csv")