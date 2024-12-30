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
    st.write("Use this section to predict consumer trends and potential sales using the trained Neural Network model.")

    try:
        # Get all feature columns from the data
        # Exclude the target variables
        feature_columns = data.columns.tolist()
        target_columns = ['binary_target', 'unit_sales(in millions)']
        feature_columns = [col for col in feature_columns if col not in target_columns]

        # Create input form
        st.subheader("Input Features")
        input_data = {}

        # Categorical features first
        categorical_features = [
            'food_category', 'food_department', 'food_family',
            'promotion_name', 'sales_country', 'marital_status', 'gender'
        ]
        
        for col in categorical_features:
            if col in data.columns:
                unique_values = sorted(data[col].unique().tolist())
                input_data[col] = st.selectbox(f"Select {col}", unique_values)

        # Numerical features
        numerical_features = [
            'store_sales(in millions)', 'store_cost(in millions)',
            'grocery_sqft', 'meat_sqft'
        ]
        
        for col in numerical_features:
            if col in data.columns:
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                mean_val = float(data[col].mean())
                
                input_data[col] = st.slider(
                    f"Select {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    format="%.2f"
                )

        # Prediction Button
        if st.button("Predict"):
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Preprocess the input data
            # Get dummy variables for categorical features
            input_processed = pd.get_dummies(input_df)
            
            # Ensure all columns from training match
            training_columns = pd.get_dummies(data[feature_columns]).columns
            
            # Add missing columns with zeros
            for col in training_columns:
                if col not in input_processed.columns:
                    input_processed[col] = 0
                    
            # Reorder columns to match training data
            input_processed = input_processed[training_columns]
            
            # Scale numerical features
            scaler = StandardScaler()
            numerical_cols = [col for col in numerical_features if col in input_processed.columns]
            if numerical_cols:
                scaler.fit(data[numerical_cols])
                input_processed[numerical_cols] = scaler.transform(input_processed[numerical_cols])

            # Verify shape matches model input
            st.write("Debug - Input shape:", input_processed.shape)
            
            # Make prediction
            try:
                prediction = model.predict(input_processed)
                prediction_class = (prediction > 0.5).astype(int)

                # Display results
                st.subheader("Prediction Results")
                st.write(f"Predicted Class: **{'Above Threshold' if prediction_class[0] == 1 else 'Below Threshold'}**")
                st.write(f"Prediction Probability: **{prediction[0][0]:.2f}**")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.write("Input shape:", input_processed.shape)
                st.write("Expected shape:", model.input_shape)

    except Exception as e:
        st.error("Error in prediction section:")
        st.error(str(e))
        st.write("Debug information:")
        st.write("Available columns:", data.columns.tolist())
        st.write("Model input shape:", model.input_shape)
# Footer
st.write("-----")
st.markdown("**Made with ❤️ for Final Year Project**")
