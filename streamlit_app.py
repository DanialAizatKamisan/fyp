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

data = load_data()
model = load_model_file()

# Identify model columns
drop_columns = ['binary_target', 'unit_sales(in millions)']
model_columns = data.drop(columns=[col for col in drop_columns if col in data.columns]).columns.tolist()

# Data Preprocessing Function
def preprocess_input(input_df, model_columns):
    """
    Preprocess input data to match the model's training features.
    - Align columns with the training dataset (one-hot encoded).
    - Scale numerical data.
    """
    # One-hot encode categorical columns in the input
    input_df = pd.get_dummies(input_df)
    
    # Align input DataFrame with the model columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale numeric data
    scaler = StandardScaler()
    scaler.fit(data[model_columns])  # Fit on training data columns
    input_scaled = scaler.transform(input_df)

    return input_scaled

# Home Section
if options == "Home":
    st.header("Overview")
    st.write("""
    This dashboard helps restaurant managers make data-driven decisions by analyzing consumer trends and predicting future outcomes. 
    Use the navigation menu to explore visualizations or make predictions with our trained Neural Network model.
    """)
    st.write("Here’s a preview of the dataset:")
    st.dataframe(data.head(10))

# Visualization Section
elif options == "Visualizations":
    st.header("Visualizations: Trends and Insights")
    st.write("Here are some key insights based on the data:")

    # Example Visualization: Unit Sales Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['unit_sales(in millions)'], kde=True, color="blue", ax=ax)
    ax.set_title("Distribution of Unit Sales")
    ax.set_xlabel("Unit Sales (in millions)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Waste vs. Unit Sales
    if "waste(in millions)" in data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=data['unit_sales(in millions)'], y=data['waste(in millions)'], ax=ax, color="orange")
        ax.set_title("Relationship Between Unit Sales and Waste")
        ax.set_xlabel("Unit Sales (in millions)")
        ax.set_ylabel("Waste (in millions)")
        st.pyplot(fig)

# Prediction Section
elif options == "Prediction":
    st.header("Make Predictions")
    st.write("Use this section to predict consumer trends and potential sales using the trained Neural Network model.")

    # Input Features
    st.subheader("Input Features")
    input_data = {}

    # Separate numerical and categorical columns
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object' and col not in drop_columns]
    numerical_columns = [col for col in data.columns if col not in drop_columns and data[col].dtype != 'object']

    # Handle categorical inputs
    for col in categorical_columns:
        unique_values = data[col].unique().tolist()
        input_data[col] = st.selectbox(f"Select {col}", unique_values)

    # Handle numerical inputs
    for col in numerical_columns:
        input_data[col] = st.slider(f"Select {col}", 
                                    float(data[col].min()), 
                                    float(data[col].max()), 
                                    float(data[col].mean()))

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess input
    try:
        input_scaled = preprocess_input(input_df, model_columns)
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        st.stop()

    # Prediction Button
    if st.button("Predict"):
        try:
            prediction = model.predict(input_scaled)
            prediction_class = (prediction > 0.5).astype(int)  # Binary classification threshold

            st.subheader("Prediction Results")
            st.write(f"Predicted Class: **{'Above Threshold' if prediction_class[0] == 1 else 'Below Threshold'}**")
            st.write(f"Prediction Probability: **{prediction[0][0]:.2f}**")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

st.write("-----")
st.markdown("**Made with ❤️ for Final Year Project**")
