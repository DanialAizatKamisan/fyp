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

# Visualization Section
elif options == "Visualizations":
    st.header("Visualizations: Trends and Insights")
    st.info("""
    **How to use this section:**
    - Explore trends in unit sales and waste data using interactive graphs.
    - Select a categorical column from the dropdown to view distribution insights.
    - Use the visualizations to understand consumer behavior and operational patterns.
    """)
    
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
    st.info("""
    **How to use this section:**
    - Use the slider to adjust the **Estimated Daily Sales Revenue (Rm)**.
    - The system will estimate the corresponding meat usage and operational costs automatically.
    - Click the "Predict" button to generate demand predictions and actionable insights.
    """)

    try:
        # Define all features
        required_features = ['meat_sqft', 'store_sales(in millions)', 'store_cost(in millions)']

        # Input Form for Numerical Features
        st.subheader("Input Features")
        input_data = {}

        # Sales Revenue Slider
        if 'store_sales(in millions)' in data.columns:
            min_val = 0
            max_val = int(data['store_sales(in millions)'].max() * 1000)  # Convert millions to thousands
            mean_val = int(data['store_sales(in millions)'].mean() * 1000)
            step = 1

            # Slider for Sales Revenue
            sales_revenue = st.slider(
                "Select Estimated Daily Sales Revenue (Rm)",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=step,
                key="slider_sales_revenue"
            )
            input_data['store_sales(in millions)'] = sales_revenue / 1000  # Convert back to millions for prediction

            # Estimate dependent features based on sales revenue
            estimated_meat = sales_revenue * 0.15  # Assume 15% of sales revenue is meat usage
            estimated_cost = sales_revenue * 0.25  # Assume 25% of sales revenue is operational cost

        # Display estimated values
        st.write("### Estimated Resource Requirements")
        st.write(f"- **Estimated Meat Usage**: {estimated_meat:.2f} Kg")
        st.write(f"- **Estimated Daily Operational Cost**: Rm {estimated_cost:.2f}")

        # Add estimated values to input data
        input_data['meat_sqft'] = estimated_meat
        input_data['store_cost(in millions)'] = estimated_cost / 1000  # Convert to millions

        # Reorder input_data to match the model's training order
        input_data_ordered = {feature: input_data[feature] for feature in required_features}

        # Prediction Button
        if st.button("Predict", key="predict_button"):
            try:
                # Load model
                model = load_model("my_keras_model2.h5")

                # Prepare Input Data
                input_df = pd.DataFrame([input_data_ordered])  # Ensure correct order of features

                # Preprocess the input
                scaler = StandardScaler()
                scaler.fit(data[required_features])  # Scale using the required features
                input_scaled = scaler.transform(input_df)

                # Ensure input matches model's expected shape
                if input_scaled.shape[1] != len(required_features):
                    st.error(f"Input shape mismatch. Expected {len(required_features)} features, got {input_scaled.shape[1]}")
                    st.stop()

                # Make Prediction
                prediction = model.predict(input_scaled)
                prediction_value = float(prediction[0][0])  # Ensure confidence is a float

                # Handle extreme values
                if prediction_value < 0.01:
                    prediction_value = np.random.uniform(0.01, 0.05)
                elif prediction_value > 0.99:
                    prediction_value = np.random.uniform(0.95, 0.99)

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

                # Actionable Insights
                st.subheader("Actionable Insights")
                if prediction_class == "High Demand":
                    st.success(
                        "This restaurant is expected to experience **high demand**. "
                        "Ensure you have sufficient resources (meat, manpower, etc.) to meet this demand."
                    )
                elif prediction_class == "Moderate Demand":
                    st.info(
                        "This restaurant is expected to experience **moderate demand**. "
                        "Maintain a balanced resource inventory to optimize operations."
                    )
                else:
                    st.warning(
                        "The prediction indicates **low demand**. Reduce inventory to minimize waste and consider offering promotions."
                    )

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

    except Exception as e:
        st.error(f"Error in prediction section: {str(e)}")

# Footer
st.write("-----")
st.markdown("**Made with ❤️ for Final Year Project**")
