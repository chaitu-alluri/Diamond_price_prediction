import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from io import StringIO
st.set_page_config(page_title="Diamond Price Prediction",page_icon="https://example.com/diamond_icon.png")
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
    diamond_emoji = "💎"
    clarity_emoji = "✨"
    depth_emoji = "🌟"
    table_emoji = "💖"
    cut_emoji = "💍"
    color_emoji = "👑"
    carat_emoji = "⚜️"
    st.title(':blue[Diamond Price Prediction App]')
    st.header('This app predicts the price of a diamond based on its characteristics.', divider='rainbow')
    st.write(diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji,diamond_emoji)

    # User input for diamond characteristics
    carat = st.slider("Carat", min_value=0.2, max_value=5.0, step=0.01, value=1.0)
    depth = st.slider("Depth", min_value=50.0, max_value=75.0, step=0.1, value=60.0)
    st.sidebar.write("CARAT:",carat_emoji)
    st.sidebar.write(
"Carat is a measure of weight used for diamonds, with one carat equal to 0.2 grams. It's often associated with the size of the diamond, where a higher carat weight generally means a larger diamond. ")
    st.sidebar.write("DEPTH:",depth_emoji)
    st.sidebar.write("Depth is a measurement of how deep the diamond is, from the table (top) to the culet (bottom). It's expressed as a percentage of the diamond's overall width. Depth affects the diamond's brilliance and sparkle, with an ideal depth percentage providing optimal light reflection and refraction. ")
    st.sidebar.write("TABLE:",table_emoji)
    st.sidebar.write("Table of a diamond is the flat surface on the top of the diamond when you look at it from above. The size of this flat surface can affect how the diamond looks and shines.", unsafe_allow_html=True)
    table = st.slider("Table", min_value=50.0, max_value=100.0, step=0.1, value=57.0)
    st.sidebar.write("CUT:",cut_emoji)
    st.sidebar.write("<u><span style='color:white; font-size:16px;'>Ideal:</span></u> Best cut, maximizes sparkle.",
                     unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>Premium:</span></u> Very good cut, still sparkles well.",
        unsafe_allow_html=True)
    st.sidebar.write("<u><span style='color:white; font-size:16px;'>Very Good:</span></u> Good cut, nice sparkle.",
                     unsafe_allow_html=True)
    st.sidebar.write("<u><span style='color:white; font-size:16px;'>Good:</span></u> Decent cut, reflects light.",
                     unsafe_allow_html=True)
    st.sidebar.write("<u><span style='color:white; font-size:16px;'>Fair:</span></u> Basic cut, less sparkle.",
                     unsafe_allow_html=True)

    cut = st.selectbox("Cut", df['cut'].unique())
    st.sidebar.write("COLOR:",color_emoji)
    st.sidebar.write("<u><span style='color:white; font-size:16px;'>D:</span></u> Colorless, highest quality.",
                     unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>E:</span></u> Near colorless, minimal tint, still high quality.",
        unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>F:</span></u> Near colorless, slight tint, still high quality.",
        unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>G:</span></u> Near colorless, noticeable tint, good quality.",
        unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>H:</span></u> Near colorless, more noticeable tint, good quality.",
        unsafe_allow_html=True)
    st.sidebar.write("<u><span style='color:white; font-size:16px;'>I:</span></u> Slightly tinted, lower quality.",
                     unsafe_allow_html=True)
    st.sidebar.write("<u><span style='color:white; font-size:16px;'>J:</span></u> Tinted, lower quality.",
                     unsafe_allow_html=True)
    color = st.selectbox("Color", df['color'].unique())
    st.sidebar.write("CLARITY:",clarity_emoji)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>IF (Internally Flawless):</span></u> No internal flaws visible under 10x magnification, extremely rare and valuable.",
        unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>VVS1 (Very, Very Slightly Included 1):</span></u> Tiny inclusions difficult to see under 10x magnification, exceptional clarity.",
        unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>VVS2 (Very, Very Slightly Included 2):</span></u> Tiny inclusions slightly more visible under 10x magnification, still very high clarity.",
        unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>VS1 (Very Slightly Included 1):</span></u> Small inclusions visible under 10x magnification, excellent clarity.",
        unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>VS2 (Very Slightly Included 2):</span></u> Slightly larger inclusions, still not visible to the naked eye, good clarity.",
        unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>SI1 (Slightly Included 1):</span></u> Noticeable inclusions under 10x magnification, but usually not visible to the naked eye, decent clarity.",
        unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>SI2 (Slightly Included 2):</span></u> Larger and more noticeable inclusions, may be visible to the naked eye, fair clarity.",
        unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>I1 (Included 1):</span></u> Inclusions visible to the naked eye, lower clarity.",
        unsafe_allow_html=True)
    st.sidebar.write(
        "<u><span style='color:white; font-size:16px;'>I2 (Included 2) and I3 (Included 3):</span></u> Obvious inclusions visible to the naked eye, lowest clarity grade, usually not recommended for jewelry.",
        unsafe_allow_html=True)
    clarity = st.selectbox("Clarity", df['clarity'].unique())
    st.divider()


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
