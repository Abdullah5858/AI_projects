import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('best_model2.h5')

# Configure page settings
st.set_page_config(page_title="Product Category Prediction", layout="wide")

# Categories predicted by the model
emoji_categories = {
    'electronics': '🔌',
    'pet_products': '🐶',
    'home_goods': '🏠',
    'stationery': '✏️',
    'fashion': '👗',
    'sports_products': '🏀',
    'cleaning': '🧹',
    'food_and_drink': '🍎'
}

def predict_category(image):
    image = image.resize((96, 96))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    category_names = list(emoji_categories.keys())
    
    return category_names[predicted_index], confidence

def main():
    st.title("🛍️ Product Category Prediction App")

    with st.sidebar:
        st.markdown(
            """
            <style>
            .css-1d391kg {
                background-color: #ADD8E6;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown("<h1 style='text-align: center;'> EMSBAY HANDEL</h1>", unsafe_allow_html=True)
        st.header("Information")
        st.write("This application predicts the product category based on the uploaded image.")
        
        if "show_info" not in st.session_state:
            st.session_state.show_info = False

        if st.button("More Information"):
            st.session_state.show_info = not st.session_state.show_info

        if st.session_state.show_info:
            st.write("# Project Details")
            st.write("""
                This project predicts product categories using various machine learning models.
                
                **Model Used:**
                - Xception (Chosen for its best score)
                
                **Goals:**
                - Increase accuracy.
                - Enhance user experience.
                - Reduce processing time.
            """)
        
        st.header("Categories")
        for category, emoji in emoji_categories.items():
            st.write(f"{emoji} {category.replace('_', ' ').capitalize()}")

        # Contact and social links
        st.write("---")
        st.write("📧 cobanabdullahgazi@gmail.com")
        st.markdown(
    """
    <a href="https://www.linkedin.com/in/abdullah-gazi-coban" target="_blank">
        <img src="https://img.icons8.com/ios-filled/50/000000/linkedin.png" style="vertical-align: middle; margin-right: 10px;"/>
        LinkedIn
    </a>
    """,
    unsafe_allow_html=True
)

    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if "predictions" not in st.session_state:
        st.session_state.predictions = {}

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width=400)

            if st.button("Predict", key=f"predict_{uploaded_file.name}"):
                category, confidence = predict_category(image)
                st.session_state.predictions[uploaded_file.name] = (category, confidence)
                st.success(f"This product category is: **{category.replace('_', ' ').capitalize()}**")
                st.info(f"Confidence: {confidence:.2f}")

                category_colors = {
                    'electronics': 'blue',
                    'pet_products': 'green',
                    'home_goods': 'orange',
                    'stationery': 'purple',
                    'fashion': 'red',
                    'sports_products': 'pink',
                    'cleaning': 'cyan',
                    'food_and_drink': 'brown'
                }
                st.markdown(
                    f"<div style='background-color: {category_colors[category]}; padding: 10px; color: white; text-align: center;'>"
                    f"Predicted Category: {category.replace('_', ' ').capitalize()}</div>",
                    unsafe_allow_html=True
                )

        if st.button("Add New Product"):
            st.session_state.predictions.clear()

if __name__ == "__main__":
    main()