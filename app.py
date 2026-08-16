import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import sklearn
st.set_page_config(page_title="Food Intelligence AI", layout="wide")

# Exact paths from root folder
csv_path = "food_demand_project/data/processed/food_sales_clean.csv"
model_path = "food_demand_project/models/best_demand_model.pkl"

# Backup checks (folder structure ela unna find chestundi)
if not os.path.exists(csv_path):
    csv_path = "data/processed/food_sales_clean.csv"
if not os.path.exists(model_path):
    model_path = "models/best_demand_model.pkl"

# Safe Model Loading
model = None
try:
    if os.path.exists(model_path):
        artifact = joblib.load(model_path)
        if isinstance(artifact, dict) and "model_pipeline" in artifact:
            model = artifact["model_pipeline"]
        else:
            model = artifact
except Exception as e:
    st.warning(f"Model load notice: {e}")
RECIPES = {
    "Chicken Biryani": {"Raw Rice": 0.25, "Chicken": 0.30, "Spices & Oil": 0.08},
    "Veg Biryani": {"Raw Rice": 0.25, "Paneer": 0.15, "Spices & Oil": 0.08},
    "Paneer Butter Masala": {"Paneer": 0.25, "Spices & Oil": 0.10},
    "Chicken Fried Rice": {"Raw Rice": 0.20, "Chicken": 0.20, "Spices & Oil": 0.05},
    "Samosa (Plate of 2)": {"Spices & Oil": 0.15}
}

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select Page", ["Executive Dashboard", "Live Prediction Demo", "Inventory Procurement", "Model Explainability"])

if menu == "Executive Dashboard":
    st.title("Executive Waste & Demand Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"₹{df['revenue'].sum():,.0f}")
    c2.metric("Total Food Sold", f"{df['quantity_sold'].sum():,} units")
    c3.metric("Total Wasted", f"{df['quantity_wasted'].sum():,} units")
    c4.metric("Waste Rate", f"{(df['quantity_wasted'].sum()/df['quantity_prepared'].sum())*100:.2f}%")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.line(df.groupby('date')['revenue'].sum().reset_index(), x='date', y='revenue', title="Daily Revenue Trend"), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(df.groupby('food_item')['quantity_wasted'].sum().reset_index(), x='food_item', y='quantity_wasted', color='food_item', title="Waste by Item"), use_container_width=True)

elif menu == "Live Prediction Demo":
    st.title("Smart Demand & Waste Prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        item = st.selectbox("Food Item", list(RECIPES.keys()))
        cust = st.slider("Expected Customers", 100, 800, 450)
        disc = st.selectbox("Discount (%)", [0, 5, 10, 15], index=1)
    with c2:
        temp = st.number_input("Temperature (°C)", value=31.0)
        rain = st.number_input("Rainfall (mm)", value=0.0)
        dow = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x], index=4)
    with c3:
        is_hol = st.selectbox("Is Holiday?", [0, 1])
        is_ex = st.selectbox("Is Exam Day?", [0, 1])
        mon = st.slider("Month", 1, 12, 8)

    if st.button("Predict Demand"):
        hist = df[df['food_item'] == item]
        inp = pd.DataFrame([{
            'food_item': item, 'customers_count': cust, 'discount': disc, 'temperature': temp,
            'rainfall': rain, 'is_weekend': 1 if dow in [5,6] else 0, 'is_holiday': is_hol,
            'is_exam_day': is_ex, 'day_of_week': dow, 'month': mon,
            'lag_1_sales': float(hist['quantity_sold'].iloc[-1]),
            'rolling_7_sales_avg': float(hist['quantity_sold'].tail(7).mean())
        }])
        pred = int(round(model.predict(inp)[0]))
        rec_prep = int(pred + np.ceil(pred * 0.02))
        exp_waste = rec_prep - pred
        cost = hist['ingredient_cost'].iloc[0]
        price = hist['selling_price'].iloc[0] * (1 - disc/100)

        st.success("Prediction Complete!")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Demand", f"{pred} portions")
        r2.metric("Recommended Prep", f"{rec_prep} portions")
        r3.metric("Expected Waste", f"{exp_waste} portions (LOW Risk)")

        f1, f2 = st.columns(2)
        f1.metric("Expected Revenue", f"₹{pred * price:,.2f}")
        f2.metric("Estimated Waste Cost", f"₹{exp_waste * cost:,.2f}")

elif menu == "Inventory Procurement":
    st.title("Raw Material Requirements")
    sel_item = st.selectbox("Target Item", list(RECIPES.keys()))
    qty = st.slider("Portions to Prepare", 50, 600, 428)
    reqs = [{"Ingredient": k, "Required (kg/l)": round(v * qty, 2)} for k, v in RECIPES[sel_item].items()]
    st.table(pd.DataFrame(reqs))

elif menu == "Model Explainability":
    st.title("Feature Importance")
    reg = model.named_steps['regressor']
    prep = model.named_steps['preprocessor']
    cats = prep.named_transformers_['cat'].get_feature_names_out(['food_item'])
    all_f = [c for c in artifact['feature_names'] if c != 'food_item'] + list(cats)
    fi_df = pd.DataFrame({'Feature': all_f, 'Importance (%)': np.round(reg.feature_importances_ * 100, 2)}).sort_values(by='Importance (%)', ascending=False)
    st.plotly_chart(px.bar(fi_df.head(8), x='Importance (%)', y='Feature', orientation='h', title="Top Influencing Features"), use_container_width=True)
