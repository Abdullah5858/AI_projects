import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('best_model5.keras')

# Configure page settings
st.set_page_config(page_title="Ürün Kategori Tahmini", layout="wide")

# Categories predicted by the model
emoji_categories = {
    'elektronik': '🔌',
    'evcil_hayvan_urunleri': '🐶',
    'ev_esyalari': '🏠',
    'kirtasiye': '✏️',
    'moda': '👗',
    'spor_urunleri': '🏀',
    'temizlik': '🧹',
    'yiyecek_icecek': '🍎'
}

def predict_category(image):
    image = image.resize((96, 96))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    
    # Debug information
    st.write("Raw Predictions:", predictions)
    category_names = list(emoji_categories.keys())
    for idx, category in enumerate(category_names):
        st.write(f"{category}: {predictions[0][idx]:.4f}")
    
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    return category_names[predicted_index], confidence

def main():
    st.title("🛍️ Ürün Kategori Tahmini Uygulaması")

    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'> EMBASSY</h1>", unsafe_allow_html=True)
        st.header("Bilgi")
        st.write("Bu uygulama, yüklenen görüntüye dayalı olarak ürün kategorisini tahmin eder.")
        
        if "show_info" not in st.session_state:
            st.session_state.show_info = False

        if st.button("Daha Fazla Bilgi"):
            st.session_state.show_info = not st.session_state.show_info

        if st.session_state.show_info:
            st.write("# Proje Detayları")
            st.write("""
                Bu proje, çeşitli makine öğrenimi modelleri kullanarak ürün kategorilerini tahmin eder.
                
                **Kullanılan Modeller:**
                - Xception
                - MobileNetV2
                - Özel CNN Modeliniz
                - ResNet50
                
                **Amaçlar:**
                - Doğruluğu artırmak.
                - Kullanıcı deneyimini geliştirmek.
                - İşleme süresini azaltmak.
            """)
        
        st.header("Kategoriler")
        for category, emoji in emoji_categories.items():
            st.write(f"{emoji} {category.replace('_', ' ').capitalize()}")

    uploaded_files = st.file_uploader("Görüntüleri yükle", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if "predictions" not in st.session_state:
        st.session_state.predictions = {}

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption='Yüklenen Görüntü', width=400)

            if st.button("Tahmin Et", key=f"predict_{uploaded_file.name}"):
                category, confidence = predict_category(image)
                st.session_state.predictions[uploaded_file.name] = (category, confidence)
                st.success(f"Bu ürün kategorisi: **{category.replace('_', ' ').capitalize()}**")
                st.info(f"Güven: {confidence:.2f}")

                category_colors = {
                    'elektronik': 'blue',
                    'evcil_hayvan_urunleri': 'green',
                    'ev_esyalari': 'orange',
                    'kirtasiye': 'purple',
                    'moda': 'red',
                    'spor_urunleri': 'pink',
                    'temizlik': 'cyan',
                    'yiyecek_icecek': 'brown'
                }
                st.markdown(
                    f"<div style='background-color: {category_colors[category]}; padding: 10px; color: white; text-align: center;'>"
                    f"Tahmin Edilen Kategori: {category.replace('_', ' ').capitalize()}</div>",
                    unsafe_allow_html=True
                )

        if st.button("Yeni Ürün Ekle"):
            st.session_state.predictions.clear()

if __name__ == "__main__":
    main()