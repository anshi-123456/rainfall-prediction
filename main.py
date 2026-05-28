import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- Page Configuration ---
st.set_page_config(page_title="Rainfall Predictor", layout="wide")

# --- 1. Data Generation (Synthetic) ---
@st.cache_data # This keeps the app fast by caching the data
def load_data():
    states = ["Maharashtra", "Kerala", "Tamil Nadu", "Rajasthan", "Punjab"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    data = []
    for state in states:
        for i, month in enumerate(months):
            # Create 10 rows per month to give the model more to learn
            for _ in range(10):
                # Logic: Higher rain in Monsoon months (Jun-Sep)
                if month in ["Jun", "Jul", "Aug", "Sep"]:
                    base_rainfall = 200 + (i * 2)
                    humidity = np.random.randint(70, 95)
                    temp = np.random.randint(24, 30)
                else:
                    base_rainfall = 10 + (i * i * 0.5)
                    humidity = np.random.randint(30, 60)
                    temp = np.random.randint(30, 40)

                # State modifiers
                if state == "Kerala": base_rainfall += 100
                elif state == "Rajasthan": base_rainfall -= 40
                
                # Add randomness/noise
                rainfall = max(0, base_rainfall + np.random.normal(0, 10))
                wind_speed = np.random.randint(5, 25)
                
                data.append([state, month, temp, humidity, wind_speed, rainfall])
    
    return pd.DataFrame(data, columns=["state", "month", "avg_temp", "humidity", "wind_speed", "rainfall_mm"])

df = load_data()

# --- 2. Machine Learning Pipeline ---
# Encoding categorical variables
df_encoded = pd.get_dummies(df, columns=["state", "month"])
X = df_encoded.drop("rainfall_mm", axis=1)
y = df_encoded["rainfall_mm"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
model = LinearRegression()
model.fit(X_train, y_train)
score = r2_score(y_test, model.predict(X_test))

# --- 3. Streamlit UI Layout ---
st.title("🌧️ Rainfall Prediction System")
st.markdown("This app uses a **Linear Regression** model to predict rainfall based on environmental factors.")

# Sidebar for Metrics
st.sidebar.header("Model Performance")
st.sidebar.metric("Accuracy (R² Score)", f"{score:.2%}")
st.sidebar.info("The model is trained on synthetic data for demonstration purposes.")

# Main Interface
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Input Parameters")
    state_input = st.selectbox("Select State", df["state"].unique())
    month_input = st.selectbox("Select Month", df["month"].unique())
    temp_input = st.slider("Temperature (°C)", 15, 50, 28)
    hum_input = st.slider("Humidity (%)", 10, 100, 70)
    wind_input = st.slider("Wind Speed (km/h)", 0, 50, 12)
    
    predict_btn = st.button("Predict Rainfall", type="primary")

with col2:
    st.subheader("Prediction Results")
    
    if predict_btn:
        # Prepare input data
        input_data = pd.DataFrame([[state_input, month_input, temp_input, hum_input, wind_input]],
                                  columns=["state", "month", "avg_temp", "humidity", "wind_speed"])
        
        # Encode input (same as training data)
        input_encoded = pd.get_dummies(input_data, columns=["state", "month"])
        
        # Align columns with training data
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        
        # Display results
        st.success(f"### Predicted Rainfall: **{prediction:.2f} mm**")
        
        # Additional visualizations
        st.markdown("---")
        st.write("**Input Summary:**")
        st.write(f"- **State:** {state_input}")
        st.write(f"- **Month:** {month_input}")
        st.write(f"- **Temperature:** {temp_input}°C")
        st.write(f"- **Humidity:** {hum_input}%")
        st.write(f"- **Wind Speed:** {wind_input} km/h")
        
        # Interpretation
        if prediction > 200:
            st.info("🌧️ Heavy rainfall expected! Be prepared.")
        elif prediction > 100:
            st.warning("☁️ Moderate rainfall predicted.")
        elif prediction > 50:
            st.info("🌦️ Light rainfall expected.")
        else:
            st.success("☀️ Minimal or no rainfall predicted.")
    else:
        st.info("👈 Adjust the parameters and click 'Predict Rainfall' to see results.")

# --- 4. Data Visualization Section ---
st.markdown("---")
st.subheader("📊 Historical Data Analysis")

tab1, tab2, tab3 = st.tabs(["Data Table", "Rainfall by State", "Rainfall by Month"])

with tab1:
    st.dataframe(df.head(50), use_container_width=True)

with tab2:
    avg_by_state = df.groupby("state")["rainfall_mm"].mean().sort_values(ascending=False)
    st.bar_chart(avg_by_state)

with tab3:
    avg_by_month = df.groupby("month")["rainfall_mm"].mean()
    # Reorder months
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    avg_by_month = avg_by_month.reindex(month_order)
    st.line_chart(avg_by_month)

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Machine Learning Model: Linear Regression")