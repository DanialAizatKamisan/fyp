# Streamlit App Code
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Title and Introduction
st.title('Final Year Project: Using Data Analytics to Identify Consumer Trends and Reduce Food Waste in Restaurants 🥗')

st.info('''
Welcome to the **Restaurant Analytics Dashboard**!
This platform uses advanced data analytics to **identify consumer trends** and provide insights to help **reduce food waste** in restaurants.

**Features:**
- View trends in consumer preferences.
- Explore key metrics related to unit sales and waste reduction.
- Predict future trends using machine learning models.

Together, we aim to promote sustainability and improve decision-making in the restaurant industry.
''')

# Sidebar Navigation with Unique Key
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Visualizations", "Prediction"], key="navigation_radio")

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
    
    # Debug information
    st.sidebar.write("Debug Info:")
    st.sidebar.write("Data shape:", data.shape)
    st.sidebar.write("Model input shape:", model.input_shape)
except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
    st.stop()

# Data Preprocessing Function
def preprocess_input(input_df, original_data):
    numerical_columns = [col for col in input_df.columns if input_df[col].dtype in ['float64', 'int64']]
    scaler = StandardScaler()
    scaler.fit(original_data[numerical_columns])
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
    return input_df

# Home Section
if options == "Home":
    st.header("Overview")
    st.write("""
    This dashboard helps restaurant managers make data-driven decisions by analyzing consumer trends and predicting future outcomes. 
    Use the navigation menu to explore visualizations or make predictions with our trained Neural Network model.
    """)
    st.write("Here's a preview of the dataset:")
    st.dataframe(data.head(10))
    st.write("Columns:", data.columns.tolist())

# Visualization Section
elif options == "Visualizations":
    st.header("Visualizations: Trends and Insights")

    if 'unit_sales(in millions)' in data.columns:
        st.subheader("Unit Sales Distribution")
        st.bar_chart(data['unit_sales(in millions)'])

# Prediction Section
elif options == "Prediction":
    st.header("Make Predictions")
    st.write("Use this section to predict consumer trends using the trained model.")

    try:
        # Define numerical features
        numerical_features = ['meat_sqft', 'store_sales(in millions)', 'store_cost(in millions)']

        # Create input form for numerical features
        st.subheader("Input Features")
        input_data = {}

        for col in numerical_features:
            if col in data.columns:
                min_val = int(data[col].min() * 1000) if 'in millions' in col else int(data[col].min())
                max_val = int(data[col].max() * 1000) if 'in millions' in col else int(data[col].max())
                mean_val = int(data[col].mean() * 1000) if 'in millions' in col else int(data[col].mean())

                # Add slider with unique key
                input_data[col] = st.slider(
                    f"Select {col.replace('(in millions)', '(in thousands)')}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=100 if 'in millions' in col else 10,
                    key=f"slider_{col}"  # Unique key for each slider
                )

        # Prediction Button
        if st.button("Predict", key="predict_button"):
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])

            # Transform slider values for prediction
            for col in ['store_sales(in millions)', 'store_cost(in millions)']:
                if col in input_df.columns:
                    input_df[col] = input_df[col] / 1000  # Convert back to millions

            # Preprocess input
            input_processed = preprocess_input(input_df, data)

            # Ensure input matches model's expected shape
            expected_shape = model.input_shape[1]
            current_shape = input_processed.shape[1]

            if current_shape < expected_shape:
                for i in range(current_shape, expected_shape):
                    input_processed[f"dummy_feature_{i}"] = 0.0

            try:
                # Make prediction
                prediction = model.predict(input_processed)
                prediction_value = prediction[0][0]
                prediction_class = "High Demand" if prediction_value > 0.5 else "Low Demand"

                # Display Results
                st.subheader("Prediction Results")
                st.write(f"Predicted Class: **{prediction_class}**")
                st.write(f"Prediction Confidence: **{prediction_value:.2f}**")

                # Actionable Insights
                if prediction_class == "High Demand":
                    st.success(
                        "Based on the prediction, this restaurant location is expected to experience **high demand**. "
                        "Consider increasing inventory for critical items to avoid stockouts."
                    )
                else:
                    st.warning(
                        "The prediction indicates **low demand**. Reduce inventory to minimize waste."
                    )

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    except Exception as e:
        st.error(f"Error in prediction section: {str(e)}")

# Footer
st.write("-----")
st.markdown("**Made with ❤️ for Final Year Project**")
