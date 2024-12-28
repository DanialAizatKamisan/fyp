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

# Data Preprocessing Function
def preprocess_input(input_df, original_df, scaler=None):
    """
    Preprocess input data to align with the training dataset structure.
    - Ensures column consistency with the training dataset.
    - Scales numeric data.
    """
    # Align input with original dataset columns
    input_df = input_df.reindex(columns=original_df.columns, fill_value=0)
    
    # Scale numeric data
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(input_df)
    else:
        scaled_data = scaler.transform(input_df)
    return scaled_data, scaler

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

    # Example Visualization: Unit Sales Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['unit_sales(in millions)'], kde=True, color="blue", ax=ax)
    ax.set_title("Distribution of Unit Sales")
    ax.set_xlabel("Unit Sales (in millions)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Example Visualization: Waste vs. Unit Sales (if applicable)
    if "waste(in millions)" in data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=data['unit_sales(in millions)'], y=data['waste(in millions)'], ax=ax, color="orange")
        ax.set_title("Relationship Between Unit Sales and Waste")
        ax.set_xlabel("Unit Sales (in millions)")
        ax.set_ylabel("Waste (in millions)")
        st.pyplot(fig)

    # Example Visualization: Top Categories (if category data exists)
    if "category" in data.columns:
        top_categories = data['category'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_categories.values, y=top_categories.index, ax=ax, palette="viridis")
        ax.set_title("Top 10 Categories by Frequency")
        ax.set_xlabel("Count")
        ax.set_ylabel("Category")
        st.pyplot(fig)

# First, let's add a function to get all possible categorical values from the training data
def get_training_feature_mapping(data):
    """Create a mapping of all possible categorical values from training data"""
    feature_mapping = {}
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    
    # Get all unique values for each categorical column
    for col in categorical_columns:
        feature_mapping[col] = sorted(data[col].unique().tolist())
    
    return feature_mapping

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
            
            # Show detailed feature information
            st.write("\nDetailed Feature Information:")
            for col in categorical_columns:
                st.write(f"\n{col}:")
                st.write(f"- Values in training data: {feature_mapping[col]}")
                st.write(f"- Generated dummy columns: {[c for c in input_processed.columns if c.startswith(col+'_')]}")
            
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
            
st.write("-----")
st.markdown("**Made with ❤️ for Final Year Project**")
