import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Used Car Price Estimator")
st.markdown("Welcome to our machine learning portal. Enter the car specifications below to instantly estimate its fair market value.")

@st.cache_resource
def load_resources():
    model = joblib.load('car_price_predictor.pkl')
    raw_data = pd.read_csv('used_cars.csv')
    return model, raw_data

try:
    model, raw_df = load_resources()
except Exception as e:
    st.error("⚠️ Critical Error: Could not find 'car_price_predictor.pkl' or 'used_cars.csv'. Please ensure they are in the same folder as this app.")
    st.stop()

st.subheader("Vehicle Specifications")

col1, col2 = st.columns(2)

with col1:
    brands = sorted(raw_df['brand'].dropna().unique().tolist())
    brand = st.selectbox("Brand", brands)

    milage = st.number_input("Mileage (mi)", min_value=-50000, value=50000, step=1000)
    if milage < 0:
        st.error("⚠️ Validation Error: Mileage cannot be negative. Please enter a valid number.")
        
    model_year = st.slider("Model Year", min_value=1990, max_value=2024, value=2015)

with col2:
    fuel_types = sorted(raw_df['fuel_type'].dropna().unique().tolist())
    fuel_type = st.selectbox("Fuel Type", fuel_types)
    
    transmissions = sorted(raw_df['transmission'].dropna().unique().tolist())
    transmission = st.selectbox("Transmission", transmissions)
    
    accidents = sorted(raw_df['accident'].dropna().unique().tolist())
    accident = st.selectbox("Accident History", accidents)
    
    titles = sorted(raw_df['clean_title'].dropna().unique().tolist())
    clean_title = st.selectbox("Title Status", titles)

st.markdown("---")

if st.button("Predict Price", type="primary"):
    if milage < 0:
        st.warning("Please fix the highlighted errors above before estimating the price.")
    else:
        with st.spinner("Calculating market value..."):
            
            user_input = pd.DataFrame({
                'model_year': [model_year],
                'milage': [milage],
                'brand': [brand],
                'fuel_type': [fuel_type],
                'transmission': [transmission],
                'accident': [accident],
                'clean_title': [clean_title]
            })
            
           
            clean_raw = raw_df.drop(['model', 'engine', 'ext_col', 'int_col', 'price'], axis=1)
            combined_df = pd.concat([clean_raw, user_input], axis=0, ignore_index=True)
            
            cols_to_encode = ['brand', 'fuel_type', 'transmission', 'accident', 'clean_title']
            combined_encoded = pd.get_dummies(combined_df, columns=cols_to_encode, drop_first=True)
            
            final_user_encoded = combined_encoded.tail(1)
        
            predicted_price = model.predict(final_user_encoded)[0]
            
            st.success(f"### 📈 Estimated Market Value: ${predicted_price:,.2f}")
            st.balloons()