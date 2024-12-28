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
''')

# Function to get all possible categorical values from the training data
def get_training_feature_mapping(data):
    """Create a mapping of all possible categorical values from training data"""
    feature_mapping = {}
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    
    # Get all unique values for each categorical column
    for col in categorical_columns:
        feature_mapping[col] = sorted(data[col].unique().tolist())
    
    return feature_mapping

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
data = load_data()
model = load_model_file()

# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Visualizations", "Prediction"])

# 1. Home Section
if options == "Home":
    st.header("Overview")
    st.write("""
    This dashboard helps restaurant managers make data-driven decisions by analyzing consumer trends 
    and predicting future outcomes. Use the navigation menu to explore visualizations or make predictions 
    with our trained Neural Network model.
    """)
    st.write("Here's a preview of the dataset:")
    st.dataframe(data.head())
    
    # Display data info
    st.write("\nDataset Information:")
    st.write(f"Total Records: {len(data)}")
    st.write(f"Total Features: {len(data.columns)}")
    
    # Display column types
    st.write("\nColumn Types:")
    for col in data.columns:
        st.write(f"- {col}: {data[col].dtype}")

# 2. Visualizations Section
elif options == "Visualizations":
    st.header("Visualizations: Trends and Insights")
    st.write("Here are some key insights based on the data:")

    # Example Visualization: Unit Sales Distribution
    if 'unit_sales(in millions)' in data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['unit_sales(in millions)'], kde=True, color="blue", ax=ax)
        ax.set_title("Distribution of Unit Sales")
        ax.set_xlabel("Unit Sales (in millions)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # Example Visualization: Waste vs. Unit Sales
    if all(col in data.columns for col in ['unit_sales(in millions)', 'waste(in millions)']):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x='unit_sales(in millions)', y='waste(in millions)', ax=ax)
        ax.set_title("Relationship Between Unit Sales and Waste")
        st.pyplot(fig)

# 3. Prediction Section
elif options == "Prediction":
    st.header("Make Predictions")
    
    # Get feature mapping from training data
    feature_mapping = get_training_feature_mapping(data)
    
    # Separate numerical and categorical columns
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    numerical_columns = [col for col in data.columns if data[col].dtype != 'object' 
                        and col not in ['binary_target', 'unit_sales(in millions)']]
    
    # Debug information
    st.write("Model expects:", model.input_shape[1], "features")
    st.write("Number of numerical columns:", len(numerical_columns))
    st.write("Number of categorical columns:", len(categorical_columns))
    
    # Display raw data structure
    st.write("\nDataset Preview:")
    st.write(data.head())
    
    # Display feature mapping
    st.write("\nFeature Mapping:")
    st.write(feature_mapping)
    
    # Input Features
    st.subheader("Input Features")
    input_data = {}
    
    # Handle categorical inputs using the training data mapping
    for col in categorical_columns:
        input_data[col] = st.selectbox(f"Select {col}", feature_mapping[col])

    # Handle numerical inputs
    for col in numerical_columns:
        input_data[col] = st.slider(f"Select {col}", 
                                  float(data[col].min()), 
                                  float(data[col].max()), 
                                  float(data[col].mean()))

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    try:
        # Create dummy variables for categorical columns using the complete set of categories
        processed_categoricals = []
        for col in categorical_columns:
            # Get dummies for current column
            dummies = pd.get_dummies(input_df[col], prefix=col)
            
            # Add missing dummy columns that were in training data
            for cat in feature_mapping[col]:
                dummy_col = f"{col}_{cat}"
                if dummy_col not in dummies.columns:
                    dummies[dummy_col] = 0
                    
            # Sort columns to ensure consistent ordering
            dummies = dummies.reindex(sorted(dummies.columns), axis=1)
            processed_categoricals.append(dummies)
        
        # Combine all categorical features
        if processed_categoricals:
            input_encoded = pd.concat(processed_categoricals, axis=1)
        else:
            input_encoded = pd.DataFrame()

        # Scale numerical columns
        if numerical_columns:
            scaler = StandardScaler()
            scaler.fit(data[numerical_columns])
            scaled_numerical_data = pd.DataFrame(
                scaler.transform(input_df[numerical_columns]),
                columns=numerical_columns
            )
            
            # Combine numerical and categorical data
            input_processed = pd.concat([scaled_numerical_data, input_encoded], axis=1)
        else:
            input_processed = input_encoded

        # Debug: Show feature counts
        st.write("\nFeature Count Summary:")
        st.write(f"- Numerical features: {len(numerical_columns)}")
        st.write(f"- Encoded categorical features: {input_encoded.shape[1]}")
        st.write(f"- Total features: {input_processed.shape[1]}")
        
        if input_processed.shape[1] != model.input_shape[1]:
            st.error(f"""
            Feature mismatch! 
            - Got {input_processed.shape[1]} features 
            - Model expects {model.input_shape[1]} features
            - Difference: {input_processed.shape[1] - model.input_shape[1]} features
            """)
            st.stop()

    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        st.write("Full error:", str(e))
        st.stop()

    # Prediction Button
    if st.button("Predict"):
        try:
            prediction = model.predict(input_processed)
            prediction_class = (prediction > 0.5).astype(int)
            st.subheader("Prediction Results")
            st.write(f"Predicted Class: **{'Above Threshold' if prediction_class[0] == 1 else 'Below Threshold'}**")
            st.write(f"Prediction Probability: **{prediction[0][0]:.2f}**")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Footer
st.write("---")
st.markdown("**Made with ❤️ for Final Year Project**")
