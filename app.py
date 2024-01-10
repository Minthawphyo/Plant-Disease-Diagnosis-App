import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

model_filepath = 'models/best_model_plant_disease_classification_noramlCNN.h5'
class_names = [
    'Pepper bell Bacterial spot', 'Pepper bell healthy',
    'Potato Early blight', 'Potato Late blight',
    'Potato healthy', 'Tomato Bacterial spot', 'Tomato Early blight',
    'Tomato Late blight', 'Tomato Leaf Mold',
    'Tomato Septoria leaf spot',
    'Tomato Spider mites Two spotted spider mite',
    'Tomato Target Spot', 'Tomato Tomato YellowLeaf Curl Virus',
    'Tomato Tomato mosaic virus', 'Tomato healthy'
]


class_info = {
    'Pepper bell Bacterial spot': {
        'title': 'Bacterial Spot on Pepper Bell',
        'sidebar_color': 'warning',
        'info': 'Bacterial spot is a common disease affecting pepper plants. Consider using copper-based fungicides for control. Remove and destroy infected plant parts to prevent further spread.'
    },
    'Pepper bell healthy': {
        'title': 'Healthy Pepper Bell',
        'sidebar_color': 'success',
        'info': 'Great news! The pepper bell appears to be healthy. Keep monitoring for any signs of disease and maintain good agricultural practices.'
    },
    'Potato Early blight': {
        'title': 'Early Blight on Potato',
        'sidebar_color': 'warning',
        'info': 'Early blight is a fungal disease that affects potato plants. To control it, consider using fungicides like chlorothalonil. Ensure proper spacing and ventilation in your potato crop to reduce humidity.'
    },
    'Potato Late blight': {
        'title': 'Late Blight on Potato',
        'sidebar_color': 'warning',
        'info': 'Late blight is a destructive disease in potatoes. Use fungicides like metalaxyl and chlorothalonil for control. Avoid overhead irrigation, as moisture promotes the spread of the disease.'
    },
    'Potato healthy': {
        'title': 'Healthy Potato',
        'sidebar_color': 'success',
        'info': 'Great news! The potato appears to be healthy. Keep monitoring for any signs of disease and maintain good agricultural practices.'
    },
    'Tomato Bacterial spot': {
        'title': 'Bacterial Spot on Tomato',
        'sidebar_color': 'warning',
        'info': 'Bacterial spot is a common bacterial disease affecting tomatoes. Use copper-based sprays and practice good garden hygiene to manage the disease.'
    },
    'Tomato Early blight': {
        'title': 'Early Blight on Tomato',
        'sidebar_color': 'warning',
        'info': 'Early blight is a fungal disease that affects tomato plants. Remove affected leaves and use fungicides like chlorothalonil to control the spread.'
    },
    'Tomato Late blight': {
        'title': 'Late Blight on Tomato',
        'sidebar_color': 'warning',
        'info': 'Late blight is a serious disease in tomatoes. Control measures include using fungicides, proper spacing, and avoiding overhead irrigation.'
    },
    'Tomato Leaf Mold': {
        'title': 'Leaf Mold on Tomato',
        'sidebar_color': 'warning',
        'info': 'Leaf mold is a fungal disease that affects tomato foliage. Increase ventilation, avoid overhead watering, and use fungicides for control.'
    },
    'Tomato Septoria leaf spot': {
        'title': 'Septoria Leaf Spot on Tomato',
        'sidebar_color': 'warning',
        'info': 'Septoria leaf spot is a common fungal disease in tomatoes. Remove infected leaves and use fungicides for effective control.'
    },
    'Tomato Spider mites Two spotted spider mite': {
        'title': 'Two-Spotted Spider Mite on Tomato',
        'sidebar_color': 'warning',
        'info': 'Spider mites can be a problem in tomatoes. Use insecticidal soap or neem oil for control. Maintain plant health to prevent infestations.'
    },
    'Tomato Target Spot': {
        'title': 'Target Spot on Tomato',
        'sidebar_color': 'warning',
        'info': 'Target spot is a fungal disease affecting tomatoes. Control measures include proper spacing, fungicides, and removing infected plant material.'
    },
    'Tomato Tomato YellowLeaf Curl Virus': {
        'title': 'Tomato Yellow Leaf Curl Virus',
        'sidebar_color': 'warning',
        'info': 'Yellow leaf curl virus is a viral disease affecting tomatoes. Use resistant varieties and control whiteflies to manage the virus.'
    },
    'Tomato Tomato mosaic virus': {
        'title': 'Tomato Mosaic Virus',
        'sidebar_color': 'warning',
        'info': 'Tomato mosaic virus is a viral disease in tomatoes. Control measures include removing infected plants and controlling aphids.'
    },
    'Tomato healthy': {
        'title': 'Healthy Tomato',
        'sidebar_color': 'success',
        'info': 'Great news! The tomato appears to be healthy. Keep monitoring for any signs of disease and maintain good agricultural practices.'
    }
}



def preprocess_image(img):
    img = img.resize((48, 48))
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model_path, img):
    img_array = preprocess_image(img)
    
    model = tf.keras.models.load_model(model_path)
    
    predictions = model.predict(img_array)
    predict_class = class_names[np.argmax(predictions[0])]  
    confidence = round(100 * np.max(predictions[0]), 2)

    return predict_class, confidence

st.set_page_config(
    page_title="Plant Disease Classification App",
    page_icon=":seedling:"
)

with st.sidebar:
    st.image('image/mg.jpg')
    st.title("Plant Disease Detection App")
    st.subheader("Detect and identify diseases in plants")

st.title('Plant Disease Diagnosis App')

st.write("""
         # Detect Plant Diseases and Get Treatment Recommendations
         This app uses deep learning to analyze plant images and identify diseases.
         """) 

with st.expander('About this app'):
    st.write("""
            - Upload an image of a diseased plant leaf
            - The model will analyze the image and predict the disease
            - You'll get a diagnosis and treatment recommendation
            - Built using TensorFlow and Streamlit
            """)

uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    eft_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image(image, caption="Uploaded Plant Image", width=300)

    prediction_class, confidence = predict(model_filepath, image)
    
    st.sidebar.error("Accuracy: " + str(confidence) + " %")

    string = "Detected Disease: " + prediction_class
    st.subheader("Model Suggestions:")

    class_info = class_info.get(prediction_class, {})
    if class_info:
        st.markdown(f"## {class_info['title']}")
        st.info(class_info['info'])
        st.sidebar.success(string)
        if 'balloons' in class_info:
            st.balloons()


st.markdown('''
# A Note From the Developer

Hi there! I'm Min Thaw Phyo(Justin) , an aspiring Ml Enginner and the developer of this plant disease diagnosis app. I built this app to put my machine learning skills to use for an important real-world problem.

As someone with a background in plant biology and horticulture, I'm passionate about leveraging technology to help farmers, gardeners, and plant enthusiasts identify and treat common plant diseases. With this app, my goal is to make plant disease diagnosis and treatment recommendations more accessible.

This is still an early version of the app, but I plan to continue improving and adding new features over time. Please reach out if you have any feedback or ideas for how I can make it more useful! I'd love to hear from you.

 And thanks for using PlantDoc!

Min Thaw Phyo(Justin) 
Plant Disease App Developer
''')
