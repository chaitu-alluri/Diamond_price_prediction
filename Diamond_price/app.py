import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from io import StringIO

# Load the data
df = pd.read_csv('./DiamondsPrices2022.csv')

# Feature Engineering
X = df[['carat', 'depth', 'table', 'cut', 'color', 'clarity']]
y = df['price']
X = pd.get_dummies(X, columns=['cut', 'color', 'clarity'])

# Train the model
model = RandomForestRegressor()
model.fit(X, y)

# Save the trained model using pickle
with open('diamond_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)


# Define the Streamlit app
def main():
    st.title(':blue[Diamond Price Prediction App By Batch 6] :sunglasses:')
    st.header('This app predicts the price of a diamond based on its characteristics.', divider='rainbow')

    # User input for diamond characteristics
    carat = st.slider("Carat", min_value=0.2, max_value=5.0, step=0.01, value=1.0)
    depth = st.slider("Depth", min_value=50.0, max_value=75.0, step=0.1, value=60.0)
    table = st.slider("Table", min_value=50.0, max_value=100.0, step=0.1, value=57.0)
    cut = st.selectbox("Cut", df['cut'].unique())
    color = st.selectbox("Color", df['color'].unique())
    clarity = st.selectbox("Clarity", df['clarity'].unique())


    # Make prediction
    if st.button("Predict"):
        # Prepare input data
        input_data = pd.DataFrame({'carat': [carat],
                                   'depth': [depth],
                                   'table': [table],
                                   'cut': [cut],
                                   'color': [color],
                                   'clarity': [clarity]})

        # Handle potential missing categorical levels
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[X.columns]  # Ensure column order is the same as during training

        # Load the model
        with open('diamond_price_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Make prediction
        prediction = loaded_model.predict(input_data)

        # Convert prediction from dollars to rupees
        prediction_rs = prediction[0] * 84

        # Display prediction
        st.success(f"The predicted price of the diamond is Rs. ₹{prediction_rs:.2f}")



# Run the app
if __name__ == '__main__':
    main()
