import streamlit as st
import pandas as pd
import requests
import io
import os

st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.markdown(
    "<h1 style='text-align: center;'>Wine Quality Predictor</h1>",
    unsafe_allow_html=True,
)
st.write("---")

if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

st.sidebar.title("Authentication")

if st.session_state.token is None:
    auth_mode = st.sidebar.selectbox("Action", ["Login", "Sign Up"])
    username_input = st.sidebar.text_input("Username", key="username_in")
    password_input = st.sidebar.text_input(
        "Password", type="password", key="password_in"
    )

    if auth_mode == "Login":
        if st.sidebar.button("Login"):
            if not username_input or not password_input:
                st.sidebar.error("Enter username and password.")
            else:
                try:
                    res = requests.post(
                        f"{API_URL}/auth/login",
                        json={"username": username_input, "password": password_input},
                    )
                    if res.status_code == 200:
                        st.session_state.token = res.json()["access_token"]
                        st.session_state.username = username_input
                        st.sidebar.success(f"Logged in as {username_input}")
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid credentials.")
                except Exception as e:
                    st.sidebar.error(f"API Error: {e}")
    else:
        if st.sidebar.button("Register"):
            if not username_input or not password_input:
                st.sidebar.error("Enter username and password.")
            else:
                try:
                    res = requests.post(
                        f"{API_URL}/auth/signup",
                        json={"username": username_input, "password": password_input},
                    )
                    if res.status_code == 200:
                        st.sidebar.success("Account created. Please login.")
                    else:
                        st.sidebar.error(res.json().get("detail", "Signup failed."))
                except Exception as e:
                    st.sidebar.error(f"API Error: {e}")
else:
    st.sidebar.success(f"User: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.session_state.username = None
        st.rerun()

if st.session_state.token is None:
    st.info("Please login to use the predictor.")

    st.write("### Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="MAE", value="0.5115")
    with col2:
        st.metric(label="RMSE", value="0.6655")
    with col3:
        st.metric(label="R² Score", value="0.4109")
    with col4:
        st.metric(label="CV RMSE", value="0.7423")

else:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}

    tabs = st.tabs(["Single Prediction", "Batch Prediction", "History", "Insights"])

    with tabs[0]:
        st.write("### Input Parameters")

        col1, col2, col3 = st.columns(3)
        with col1:
            wine_type = st.selectbox("Type", ["Red", "White"])
            fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4, 0.1)
            volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.3, 0.01)
            citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3, 0.01)
        with col2:
            residual_sugar = st.slider("Residual Sugar", 0.5, 20.0, 2.0, 0.1)
            chlorides = st.slider("Chlorides", 0.01, 0.6, 0.05, 0.001)
            free_sulfur = st.slider("Free Sulfur Dioxide", 1.0, 100.0, 30.0, 1.0)
            total_sulfur = st.slider("Total Sulfur Dioxide", 5.0, 300.0, 115.0, 1.0)
        with col3:
            density = st.number_input(
                "Density", 0.9850, 1.0100, 0.9950, 0.0001, format="%.4f"
            )
            ph = st.slider("pH", 2.7, 4.0, 3.2, 0.01)
            sulphates = st.slider("Sulphates", 0.2, 2.0, 0.6, 0.01)
            alcohol = st.slider("Alcohol Vol %", 8.0, 15.0, 10.5, 0.1)

        if st.button("Predict"):
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
                "type": wine_type.lower(),
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
                        st.metric("Predicted Quality", f"{pred:.2f}")
                    with res_col2:
                        st.metric("Rounded Score", str(rounded))
                else:
                    st.error("Error from API.")
            except Exception as e:
                st.error(f"API connection failed: {e}")

    with tabs[1]:
        st.write("### CSV Upload")
        uploaded_file = st.file_uploader("Upload dataset", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            if st.button("Run Batch"):
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.read(), "text/csv")}

                try:
                    res = requests.post(
                        f"{API_URL}/predict_batch", files=files, headers=headers
                    )
                    if res.status_code == 200:
                        pred_df = pd.read_csv(io.BytesIO(res.content))
                        st.dataframe(pred_df.head())

                        csv_buffer = io.BytesIO()
                        pred_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Download Output",
                            data=csv_buffer.getvalue(),
                            file_name="output.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error("API error during batch processing.")
                except Exception as e:
                    st.error(f"API connection failed: {e}")

    with tabs[2]:
        st.write("### Prediction History")
        if st.button("Refresh"):
            st.rerun()

        try:
            res = requests.get(f"{API_URL}/predictions_history", headers=headers)
            if res.status_code == 200:
                history_logs = res.json()
                if history_logs:
                    df_logs = pd.DataFrame(history_logs)
                    df_logs["timestamp"] = pd.to_datetime(
                        df_logs["timestamp"]
                    ).dt.strftime("%Y-%m-%d %H:%M:%S")
                    st.dataframe(df_logs, use_container_width=True)
                else:
                    st.write("No records found.")
            else:
                st.error("Failed to retrieve history.")
        except Exception as e:
            st.error(f"Error fetching logs: {e}")

    with tabs[3]:
        st.write("### Features")
        features = [
            "alcohol",
            "volatile acidity",
            "free sulfur dioxide",
            "sulphates",
            "density",
            "citric acid",
            "total sulfur dioxide",
            "residual sugar",
            "chlorides",
            "pH",
            "fixed acidity",
            "is_red",
        ]
        importances = [
            0.35,
            0.15,
            0.08,
            0.07,
            0.06,
            0.05,
            0.05,
            0.05,
            0.04,
            0.04,
            0.03,
            0.03,
        ]
        chart_df = pd.DataFrame(
            {"Feature": features, "Importance": importances}
        ).sort_values(by="Importance", ascending=True)
        st.bar_chart(data=chart_df, x="Feature", y="Importance")
