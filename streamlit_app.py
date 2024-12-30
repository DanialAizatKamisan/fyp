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
    """
    Preprocess input data to match the model's training features.
    """
    # Get columns to process (exclude target variables)
    drop_columns = ['binary_target', 'unit_sales(in millions)']
    feature_columns = [col for col in original_data.columns if col not in drop_columns]
    
    # Create a copy of input data with only feature columns
    processed_df = input_df[feature_columns].copy()
    
    # One-hot encode categorical columns
    categorical_columns = processed_df.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        # Get dummy variables for both input and original data
        processed_df = pd.get_dummies(processed_df, columns=categorical_columns)
        original_dummies = pd.get_dummies(original_data[categorical_columns])
        
        # Ensure all columns from original data are present
        for col in original_dummies.columns:
            if col not in processed_df.columns:
                processed_df[col] = 0
        
        # Keep only the columns that were in the original data
        processed_df = processed_df[original_dummies.columns]
    
    # Scale numerical features
    numerical_columns = [col for col in feature_columns if col not in categorical_columns]
    if numerical_columns:
        scaler = StandardScaler()
        scaler.fit(original_data[numerical_columns])
        processed_df[numerical_columns] = scaler.transform(processed_df[numerical_columns])
    
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
    
    # Display column information
    st.write("\nColumn Information:")
    for col in data.columns:
        st.write(f"- {col}: {data[col].dtype}")

# Visualization Section
elif options == "Visualizations":
    st.header("Visualizations: Trends and Insights")
    
    # Sales Distribution
    if 'unit_sales(in millions)' in data.columns:
        st.subheader("Unit Sales Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['unit_sales(in millions)'], kde=True, color="blue", ax=ax)
        ax.set_title("Distribution of Unit Sales")
        ax.set_xlabel("Unit Sales (in millions)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # Waste vs Sales Relationship
    if all(col in data.columns for col in ['unit_sales(in millions)', 'waste(in millions)']):
        st.subheader("Sales vs Waste Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x='unit_sales(in millions)', y='waste(in millions)', ax=ax)
        ax.set_title("Relationship Between Unit Sales and Waste")
        st.pyplot(fig)

    # Categorical Analysis
    categorical_columns = data.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        st.subheader("Categorical Data Analysis")
        selected_cat = st.selectbox("Select Category to Analyze:", categorical_columns)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        data[selected_cat].value_counts().plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {selected_cat}")
        st.pyplot(fig)

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

        # Add sliders for numerical inputs starting from zero
        for col in numerical_features:
            if col in data.columns:
                if 'in millions' in col:
                    min_val = 0  # Start slider from zero
                    max_val = int(data[col].max() * 1000)  # Convert millions to thousands
                    mean_val = int(data[col].mean() * 1000)
                    step = 100  # Step by hundreds
                else:
                    min_val = 0  # Start slider from zero
                    max_val = int(data[col].max())
                    mean_val = int(data[col].mean())
                    step = 10  # Step by tens

                # Add slider
                input_data[col] = st.slider(
                    f"Select {col.replace('(in millions)', '(in thousands)')}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=step,
                    key=f"slider_{col}"  # Unique key for each slider
                )

        # Prediction Button
        if st.button("Predict", key="predict_button"):
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])

            # Transform slider values back to original scale for prediction
            for col in ['store_sales(in millions)', 'store_cost(in millions)']:
                if col in input_df.columns:
                    input_df[col] = input_df[col] / 1000  # Convert thousands back to millions

            # Debugging: Log the raw input data
            st.write("Debug - Raw Input Data:", input_df)

            # Preprocess input
            numerical_columns = [col for col in numerical_features if col in data.columns]
            scaler = StandardScaler()
            scaler.fit(data[numerical_columns])  # Fit the scaler on the original data
            input_processed = scaler.transform(input_df)

            # Convert scaled data to DataFrame
            input_processed = pd.DataFrame(input_processed, columns=numerical_columns)

            # Debugging: Log the scaled input data
            st.write("Debug - Scaled Input Data:", input_processed)

            # Ensure input matches model's expected shape
            expected_shape = 298
            current_shape = input_processed.shape[1]

            if current_shape < expected_shape:
                for i in range(current_shape, expected_shape):
                    col_name = f"dummy_feature_{i}"
                    input_processed[col_name] = 0.0

            # Debugging: Log the final input shape
            st.write("Debug - Final Input Shape:", input_processed.shape)

            try:
                # Make prediction
                prediction = model.predict(input_processed)
                prediction_value = prediction[0][0]
                prediction_class = "High Demand" if prediction_value > 0.5 else "Low Demand"

                # Display results
                st.subheader("Prediction Results")
                st.write(f"Predicted Class: **{prediction_class}**")
                st.write(f"Prediction Confidence: **{prediction_value:.2f}**")

                # Actionable Insights
                st.subheader("Actionable Insights")
                if prediction_class == "High Demand":
                    st.success(
                        "Based on the prediction, this restaurant location is expected to experience **high demand**. "
                        "Consider increasing inventory for critical items to avoid stockouts. Focus on optimizing sales of high-performing categories."
                    )
                else:
                    st.warning(
                        "The prediction indicates **low demand**. Reduce inventory to minimize waste, and consider offering promotions to boost sales."
                    )

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    except Exception as e:
        st.error(f"Error in prediction section: {str(e)}")

# Footer
st.write("-----")
st.markdown("**Made with ❤️ for Final Year Project**")
