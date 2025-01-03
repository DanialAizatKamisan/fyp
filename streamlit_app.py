import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Title and Introduction
st.title('Final Year Project: Leveraging Data Analytics to Optimize Consumer Trends and Minimize Food Waste in Restaurants 🥗')

st.info('''
Welcome to the **Restaurant Analytics Dashboard**!
This innovative platform harnesses the power of data analytics to **analyze consumer trends** and deliver actionable insights to **reduce food waste** effectively in the restaurant industry.

**Features:**
- Uncover patterns in consumer behavior and preferences.
- Analyze key metrics such as sales performance and operational costs.
- Forecast future trends using a trained neural network model.

This project aims to promote **sustainability** while enhancing **decision-making processes** for restaurant managers and stakeholders.
''')

# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Visualizations", "Prediction"])

# Load the Cleaned Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_dataset.csv")
    return data

# Load the Model
@st.cache_resource
def load_model_file():
    model = load_model("my_keras_model.h5")
    return model

# Load data and model
try:
    data = load_data()
    model = load_model_file()
except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
    st.stop()

# Data Preprocessing Function
def preprocess_input(input_df, original_data):
    feature_columns = ['meat_sqft', 'store_sales(in millions)', 'store_cost(in millions)']
    processed_df = input_df.copy()
    scaler = StandardScaler()
    scaler.fit(original_data[feature_columns])
    processed_df[feature_columns] = scaler.transform(processed_df[feature_columns])
    return processed_df

# Home Section
if options == "Home":
    st.header("Overview")
    st.write("""
    This dashboard helps restaurant managers make data-driven decisions by analyzing consumer trends and predicting future outcomes. 
    Use the navigation menu to explore visualizations or make predictions with our trained Neural Network model.
    """)
    st.write("Here's a preview of the dataset:")
    st.dataframe(data.head(10))

# Visualization Section
elif options == "Visualizations":
    st.header("Visualizations: Trends and Insights")
    st.write("Use this section to explore trends and insights from the dataset.")

    # Sales Distribution
    st.subheader("Unit Sales Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['unit_sales(in millions)'], kde=True, color="blue", ax=ax)
    ax.set_title("Distribution of Unit Sales")
    ax.set_xlabel("Unit Sales (in millions)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Prediction Section
elif options == "Prediction":
    st.header("Make Predictions")
    st.info("""
    **How to use this section:**
    - Use the slider to adjust the **Estimated Daily Sales Revenue (Rm)**.
    - The system will estimate the corresponding meat usage and operational costs automatically.
    - Click the "Predict" button to generate demand predictions and actionable insights.
    """)

    try:
        required_features = ['meat_sqft', 'store_sales(in millions)', 'store_cost(in millions)']
        input_data = {}

        # Sales Revenue Slider
        sales_revenue = st.slider(
            "Select Estimated Daily Sales Revenue (Rm)",
            min_value=0,
            max_value=30000,
            value=10000,
            step=1000
        )
        input_data['store_sales(in millions)'] = sales_revenue / 1000
        estimated_meat = sales_revenue * 0.15
        estimated_cost = sales_revenue * 0.25

        # Display estimated values in a styled box
        st.markdown(
            f"""
            <div style="
                border: 1px solid #E1E1E1; 
                border-radius: 8px; 
                padding: 16px; 
                background-color: #F9F9F9;
                margin-bottom: 16px;">
                <p style="font-size: 16px; margin: 0;">
                    <b>Estimated Meat Usage:</b> {estimated_meat:.2f} Kg
                </p>
                <p style="font-size: 16px; margin: 0;">
                    <b>Estimated Daily Operational Cost:</b> Rm {estimated_cost:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        input_data['meat_sqft'] = estimated_meat
        input_data['store_cost(in millions)'] = estimated_cost / 1000

        # Prediction Button
        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([input_data])
                scaler = StandardScaler()
                scaler.fit(data[required_features])
                input_scaled = scaler.transform(input_df)

                prediction = model.predict(input_scaled)
                prediction_value = float(prediction[0][0])

                # Determine prediction class
                if prediction_value < 0.4:
                    prediction_class = "Low Demand"
                elif 0.4 <= prediction_value <= 0.7:
                    prediction_class = "Moderate Demand"
                else:
                    prediction_class = "High Demand"

                # Display Results
                st.subheader("Prediction Results")
                st.write(f"Predicted Class: **{prediction_class}**")
                st.write(f"Prediction Confidence: **{prediction_value:.4f}**")

                # Add gauge visualization
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh([0], [prediction_value], color="green" if prediction_class == "High Demand" else "orange" if prediction_class == "Moderate Demand" else "red", height=0.5)
                ax.set_xlim(0, 1)
                ax.set_title("Prediction Confidence")
                ax.set_yticks([])
                st.pyplot(fig)

                # Actionable Insights
                st.subheader("Actionable Insights")
                if prediction_class == "High Demand":
                    st.success("Ensure sufficient resources to meet the high demand.")
                elif prediction_class == "Moderate Demand":
                    st.info("Balance resources cautiously to handle moderate demand.")
                else:
                    st.warning("Reduce inventory and consider promotions to boost sales.")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

    except Exception as e:
        st.error(f"Error in prediction section: {str(e)}")

# Footer
st.write("-----")
st.markdown("**Made with ❤️ for Final Year Project**")
