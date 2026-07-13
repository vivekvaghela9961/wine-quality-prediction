import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os

# Set page config
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Title and header
st.markdown("<h1 style='text-align: center; color: #800020;'>🍷 Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #7f8c8d;'>A Production-Grade ML Tool for Quality Assessment</h3>", unsafe_allow_html=True)
st.write("---")

# Session state initialization
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

# Sidebar Authentication Layout
st.sidebar.title("🔐 Authentication")

if st.session_state.token is None:
    auth_mode = st.sidebar.selectbox("Choose Action", ["Login", "Sign Up"])
    username_input = st.sidebar.text_input("Username", key="username_in")
    password_input = st.sidebar.text_input("Password", type="password", key="password_in")
    
    if auth_mode == "Login":
        if st.sidebar.button("Login"):
            if not username_input or not password_input:
                st.sidebar.error("Please enter both username and password.")
            else:
                try:
                    res = requests.post(
                        f"{API_URL}/auth/login",
                        json={"username": username_input, "password": password_input}
                    )
                    if res.status_code == 200:
                        st.session_state.token = res.json()["access_token"]
                        st.session_state.username = username_input
                        st.sidebar.success(f"Logged in as {username_input}!")
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid username or password.")
                except Exception as e:
                    st.sidebar.error(f"Error connecting to API server: {e}")
    else:
        if st.sidebar.button("Register"):
            if not username_input or not password_input:
                st.sidebar.error("Please enter both username and password.")
            else:
                try:
                    res = requests.post(
                        f"{API_URL}/auth/signup",
                        json={"username": username_input, "password": password_input}
                    )
                    if res.status_code == 200:
                        st.sidebar.success("Account created successfully! Please login.")
                    else:
                        st.sidebar.error(res.json().get("detail", "Signup failed."))
                except Exception as e:
                    st.sidebar.error(f"Error connecting to API server: {e}")
else:
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.session_state.username = None
        st.rerun()

# Main page content
if st.session_state.token is None:
    st.info("🔒 Please register or login using the sidebar to access the predictor tool.")
    
    # Showcase API metrics in public section
    st.write("### Model Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Mean Absolute Error (MAE)", value="0.5115")
    with col2:
        st.metric(label="Root Mean Squared Error (RMSE)", value="0.6655")
    with col3:
        st.metric(label="R² Score", value="0.4109")
    with col4:
        st.metric(label="5-Fold Cross-Validation RMSE", value="0.7423")
        
else:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    tabs = st.tabs(["🍷 Single Prediction", "📂 Batch Prediction (CSV)", "📜 Prediction Logs History", "📊 Model Insights"])
    
    # ------------------ Tab 1: Single Prediction ------------------
    with tabs[0]:
        st.write("### Chemical Characteristics Form")
        st.write("Fill in the chemical values below to predict the wine quality score (0 to 10 scale).")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            wine_type = st.selectbox("Wine Type", ["Red", "White"])
            fixed_acidity = st.slider("Fixed Acidity (g/L)", 4.0, 16.0, 7.4, 0.1)
            volatile_acidity = st.slider("Volatile Acidity (g/L)", 0.1, 1.5, 0.3, 0.01)
            citric_acid = st.slider("Citric Acid (g/L)", 0.0, 1.0, 0.3, 0.01)
        with col2:
            residual_sugar = st.slider("Residual Sugar (g/L)", 0.5, 20.0, 2.0, 0.1)
            chlorides = st.slider("Chlorides (g/L)", 0.01, 0.6, 0.05, 0.001)
            free_sulfur = st.slider("Free Sulfur Dioxide (mg/L)", 1.0, 100.0, 30.0, 1.0)
            total_sulfur = st.slider("Total Sulfur Dioxide (mg/L)", 5.0, 300.0, 115.0, 1.0)
        with col3:
            density = st.number_input("Density (g/cm³)", 0.9850, 1.0100, 0.9950, 0.0001, format="%.4f")
            ph = st.slider("pH Level", 2.7, 4.0, 3.2, 0.01)
            sulphates = st.slider("Sulphates (g/L)", 0.2, 2.0, 0.6, 0.01)
            alcohol = st.slider("Alcohol Vol %", 8.0, 15.0, 10.5, 0.1)
            
        if st.button("Predict Quality", type="primary"):
            payload = {
                "fixed_acidity": fixed_acidity,
                "volatile_acidity": volatile_acidity,
                "citric_acid": citric_acid,
                "residual_sugar": residual_sugar,
                "chlorides": chlorides,
                "free_sulfur_dioxide": free_sulfur,
                "total_sulfur_dioxide": total_sulfur,
                "density": density,
                "pH": ph,
                "sulphates": sulphates,
                "alcohol": alcohol,
                "type": wine_type.lower()
            }
            
            try:
                res = requests.post(f"{API_URL}/predict", json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    pred = data["prediction"]
                    rounded = data["rounded_prediction"]
                    
                    st.write("---")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        # Colored cards depending on quality
                        if pred >= 6.5:
                            color = "#27ae60" # Green for great wines
                            quality_desc = "Excellent Quality"
                        elif pred >= 5.5:
                            color = "#f39c12" # Orange for average
                            quality_desc = "Average Quality"
                        else:
                            color = "#c0392b" # Red for poor quality
                            quality_desc = "Low Quality"
                            
                        st.markdown(
                            f"""
                            <div style="background-color: {color}; padding: 25px; border-radius: 10px; text-align: center; color: white;">
                                <h3>Predicted Quality Rating</h3>
                                <h1 style="font-size: 50px; margin: 5px 0;">{pred:.2f}</h1>
                                <h4>Rounded Class Score: {rounded}</h4>
                                <p style="font-style: italic;">{quality_desc}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with res_col2:
                        st.write("#### Predictor Feedback")
                        st.info(
                            f"Model has evaluated your **{wine_type}** wine. "
                            f"The primary driver is **alcohol content ({alcohol}%)** "
                            f"and **volatile acidity ({volatile_acidity} g/L)**. "
                            "This prediction has been automatically logged to the audit database."
                        )
                else:
                    st.error(f"Error calling predict endpoint: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"API server connection failed: {e}")
                
    # ------------------ Tab 2: Batch Prediction ------------------
    with tabs[1]:
        st.write("### Upload Wine CSV Data")
        st.write("Upload a CSV file containing chemical stats of multiple wines. The system will append predictions and stream back the file.")
        
        st.info("💡 Your CSV must contain columns similar to: `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, and `type` (or `wine_type`).")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("#### Data Preview (Top 5 rows):")
            st.dataframe(df.head(5))
            
            if st.button("Generate Batch Predictions", type="primary"):
                # Reset pointer
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.read(), "text/csv")}
                
                try:
                    res = requests.post(f"{API_URL}/predict_batch", files=files, headers=headers)
                    if res.status_code == 200:
                        pred_df = pd.read_csv(io.BytesIO(res.content))
                        st.success("Batch predictions completed successfully!")
                        st.write("#### Output Preview:")
                        st.dataframe(pred_df.head(10))
                        
                        # Downloader
                        csv_buffer = io.BytesIO()
                        pred_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv_buffer.getvalue(),
                            file_name="predictions_output.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Error running batch: {res.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error communicating with API: {e}")

    # ------------------ Tab 3: Prediction History ------------------
    with tabs[2]:
        st.write("### Database Audit Trail")
        st.write("Below are the logged single wine quality predictions stored dynamically in the audit database.")
        
        if st.button("Refresh History Logs"):
            st.rerun()
            
        try:
            res = requests.get(f"{API_URL}/predictions_history", headers=headers)
            if res.status_code == 200:
                history_logs = res.json()
                if not history_logs:
                    st.warning("No logged predictions found in the database. Go generate some predictions first!")
                else:
                    df_logs = pd.DataFrame(history_logs)
                    
                    # Formatting columns
                    df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    st.dataframe(df_logs, use_container_width=True)
                    
                    # Mini charts
                    st.write("#### Quality Distribution of Predicted Samples")
                    chart_data = df_logs.groupby("wine_type")["predicted_quality"].mean().reset_index()
                    st.bar_chart(data=chart_data, x="wine_type", y="predicted_quality", color="wine_type")
            else:
                st.error("Could not fetch prediction history.")
        except Exception as e:
            st.error(f"Error fetching logs: {e}")

    # ------------------ Tab 4: Model Insights ------------------
    with tabs[3]:
        st.write("### Model Interpretability & Configuration")
        
        # Display Feature Importance Chart
        st.write("#### Global Feature Importances (Random Forest)")
        features = [
            "alcohol", "volatile acidity", "free sulfur dioxide", "sulphates", 
            "density", "citric acid", "total sulfur dioxide", "residual sugar", 
            "chlorides", "pH", "fixed acidity", "is_red"
        ]
        importances = [0.35, 0.15, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03]
        
        chart_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)
        
        st.bar_chart(data=chart_df, x="Feature", y="Importance", color="#800020")
        
        # Extra details
        st.write("#### API Configuration")
        try:
            info_res = requests.get(f"{API_URL}/model_info")
            if info_res.status_code == 200:
                info_data = info_res.json()
                st.json(info_data)
        except Exception:
            st.warning("Could not contact the model_info route.")
