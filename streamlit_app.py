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
        
# Prediction Section with New Page for Three Sliders
elif options == "Prediction":
    st.header("Prediction")
    st.info("""
    This section offers two prediction modes:
    - **Single Input Prediction**: Adjust a single slider to predict demand.
    - **Multi-Input Prediction**: Adjust three sliders simultaneously to predict demand.
    """)

    # Sub-navigation for prediction modes
    prediction_mode = st.radio("Choose a prediction mode:", ["Single Input", "Multi-Input"])

    if prediction_mode == "Single Input":
        st.subheader("Single Input Prediction")
        st.info("""
        Use the slider below to adjust the **Estimated Daily Sales Revenue (Rm)**.
        The model will predict the corresponding demand class and confidence score.
        """)

        # Slider for Sales Revenue
        if 'store_sales(in millions)' in data.columns:
            min_val = 0
            max_val = int(data['store_sales(in millions)'].max() * 1000)  # Convert millions to thousands
            mean_val = int(data['store_sales(in millions)'].mean() * 1000)

            # Slider for Sales Revenue
            sales_revenue = st.slider(
                "Select Estimated Daily Sales Revenue (Rm)",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=1,
                key="slider_sales_revenue"
            )

            # Prepare Input Data for Single Prediction
            input_data_single = {
                'store_sales(in millions)': sales_revenue / 1000,  # Convert back to millions
                'meat_sqft': data['meat_sqft'].mean(),
                'store_cost(in millions)': data['store_cost(in millions)'].mean()
            }

            # Prediction Button for Single Input
            if st.button("Predict for Single Input", key="predict_single"):
                try:
                    # Prepare Input Data
                    input_df = pd.DataFrame([input_data_single])

                    # Preprocess Input Data
                    scaler = StandardScaler()
                    scaler.fit(data[required_features])
                    input_scaled = scaler.transform(input_df)

                    # Make Prediction
                    prediction = model.predict(input_scaled)
                    prediction_value = float(prediction[0][0])

                    # Handle Extreme Values
                    prediction_value = np.clip(prediction_value, 0.01, 0.99)

                    # Determine Prediction Class
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

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

    elif prediction_mode == "Multi-Input":
        st.subheader("Multi-Input Prediction")
        st.info("""
        Use the sliders below to adjust **Sales Revenue (Rm)**, **Meat Usage (Kg)**, 
        and **Operational Cost (Rm)** simultaneously. 
        The model will predict the corresponding demand class and confidence score.
        """)

        # Sliders for Multiple Inputs
        input_data_multi = {}

        # Slider for Sales Revenue
        if 'store_sales(in millions)' in data.columns:
            sales_revenue_min = 0
            sales_revenue_max = int(data['store_sales(in millions)'].max() * 1000)  # Convert millions to thousands
            sales_revenue_mean = int(data['store_sales(in millions)'].mean() * 1000)

            sales_revenue = st.slider(
                "Estimated Daily Sales Revenue (Rm)",
                min_value=sales_revenue_min,
                max_value=sales_revenue_max,
                value=sales_revenue_mean,
                step=1,
                key="slider_sales_revenue_multi"
            )
            input_data_multi['store_sales(in millions)'] = sales_revenue / 1000  # Convert back to millions

        # Slider for Meat Usage
        if 'meat_sqft' in data.columns:
            meat_usage_min = 0
            meat_usage_max = int(data['meat_sqft'].max())
            meat_usage_mean = int(data['meat_sqft'].mean())

            meat_usage = st.slider(
                "Estimated Meat Usage (Kg)",
                min_value=meat_usage_min,
                max_value=meat_usage_max,
                value=meat_usage_mean,
                step=1,
                key="slider_meat_usage_multi"
            )
            input_data_multi['meat_sqft'] = meat_usage

        # Slider for Operational Cost
        if 'store_cost(in millions)' in data.columns:
            operational_cost_min = 0
            operational_cost_max = int(data['store_cost(in millions)'].max() * 1000)  # Convert millions to thousands
            operational_cost_mean = int(data['store_cost(in millions)'].mean() * 1000)

            operational_cost = st.slider(
                "Estimated Daily Operational Cost (Rm)",
                min_value=operational_cost_min,
                max_value=operational_cost_max,
                value=operational_cost_mean,
                step=1,
                key="slider_operational_cost_multi"
            )
            input_data_multi['store_cost(in millions)'] = operational_cost / 1000  # Convert back to millions

        # Display Input Data
        st.write("### Input Data for Multi-Input Prediction")
        st.write(f"- **Estimated Daily Sales Revenue (Rm):** {sales_revenue}")
        st.write(f"- **Estimated Meat Usage (Kg):** {meat_usage}")
        st.write(f"- **Estimated Daily Operational Cost (Rm):** {operational_cost}")

        # Prediction Button for Multi-Input
        if st.button("Predict for Multi-Input", key="predict_multi"):
            try:
                # Prepare Input Data
                input_df_multi = pd.DataFrame([input_data_multi])

                # Preprocess Input Data
                scaler = StandardScaler()
                scaler.fit(data[required_features])
                input_scaled_multi = scaler.transform(input_df_multi)

                # Make Prediction
                prediction_multi = model.predict(input_scaled_multi)
                prediction_value_multi = float(prediction_multi[0][0])

                # Handle Extreme Values
                prediction_value_multi = np.clip(prediction_value_multi, 0.01, 0.99)

                # Determine Prediction Class
                if prediction_value_multi < 0.4:
                    prediction_class_multi = "Low Demand"
                elif 0.4 <= prediction_value_multi <= 0.7:
                    prediction_class_multi = "Moderate Demand"
                else:
                    prediction_class_multi = "High Demand"

                # Display Results
                st.subheader("Prediction Results")
                st.write(f"Predicted Class: **{prediction_class_multi}**")
                st.write(f"Prediction Confidence: **{prediction_value_multi:.4f}**")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")


# Footer
st.write("-----")
st.markdown("**Made with ❤️ for Final Year Project**")
