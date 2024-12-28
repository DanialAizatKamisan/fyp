import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

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

# Load Dataset and Model
data = load_data()
model = load_model_file()

# Extract relevant columns for prediction
model_columns = data.drop(columns=['binary_target', 'unit_sales(in millions)']).columns.tolist()

# Data Preprocessing Function
def preprocess_input(input_df):
    """
    Align input data with the model's expected columns and scale numeric features.
    """
    # Reindex input data to match model's training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale the input data
    scaler = StandardScaler()
    scaler.fit(data[model_columns])
    scaled_input = scaler.transform(input_df)
    return scaled_input

# 1. Home Section
if options == "Home":
    st.header("Overview")
    st.write("""
    This dashboard helps restaurant managers make data-driven decisions by analyzing consumer trends and predicting future outcomes. 
    Use the navigation menu to explore visualizations or make predictions with our trained Neural Network model.
    """)
    st.write("Here’s a preview of the dataset:")
    st.dataframe(data.head(10))

# 2. Visualizations Section
elif options == "Visualizations":
    st.header("Visualizations: Trends and Insights")
    st.write("Here are some key insights based on the data:")

    # Visualization 1: Unit Sales Distribution
    if 'unit_sales(in millions)' in data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['unit_sales(in millions)'], kde=True, color="blue", ax=ax)
        ax.set_title("Distribution of Unit Sales")
        ax.set_xlabel("Unit Sales (in millions)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # Visualization 2: Waste vs. Unit Sales (if applicable)
    if 'waste(in millions)' in data.columns and 'unit_sales(in millions)' in data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=data['unit_sales(in millions)'], y=data['waste(in millions)'], ax=ax, color="orange")
        ax.set_title("Relationship Between Unit Sales and Waste")
        ax.set_xlabel("Unit Sales (in millions)")
        ax.set_ylabel("Waste (in millions)")
        st.pyplot(fig)

# 3. Prediction Section
elif options == "Prediction":
    st.header("Make Predictions")
    st.write("Use this section to predict consumer trends and potential sales using the trained Neural Network model.")

    # Input Features
    st.subheader("Input Features")
    input_data = {}

    # Display only relevant columns for prediction
    for col in model_columns:
        if data[col].dtype == 'object':  # Handle categorical columns
            unique_values = data[col].unique().tolist()
            input_data[col] = st.selectbox(f"Select {col}", unique_values)
        else:  # Handle numerical columns
            input_data[col] = st.slider(f"Select {col}", 
                                        float(data[col].min()), 
                                        float(data[col].max()), 
                                        float(data[col].mean()))

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    try:
        # Preprocess the input data
        input_scaled = preprocess_input(input_df)

        # Prediction Button
        if st.button("Predict"):
            prediction = model.predict(input_scaled)
            prediction_class = (prediction > 0.5).astype(int)  # Binary classification threshold

            st.subheader("Prediction Results")
            st.write(f"Predicted Class: **{'Above Threshold' if prediction_class[0] == 1 else 'Below Threshold'}**")
            st.write(f"Prediction Probability: **{prediction[0][0]:.2f}**")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

st.write("-----")
st.markdown("**Made with ❤️ for Final Year Project**")
