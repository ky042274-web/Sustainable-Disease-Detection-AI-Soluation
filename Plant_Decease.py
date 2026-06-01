# 🌿 AI PLANT DISEASE DETECTION SYSTEM


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import base64

# PAGE CONFIG

st.set_page_config(
    page_title="Sustainable Disease Detection AI System",
    layout="wide",
    page_icon="🌿"
)
# BACKGROUND IMAGE

def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_base64("background.png")

page_bg = f"""
<style>

.stApp {{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

h1,h2,h3,h4,p,label,div {{
    color: white !important;
}}

.title-box {{
    background: rgba(0,0,0,0.65);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
}}

.stButton>button {{
    background-color: green;
    color: white;
    border-radius: 10px;
}}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# TITLE

st.markdown("""
<div class="title-box">
<h1>🌿 Sustainable Disease Detection AI System</h1>
<p>AI Based Plant Disease Detection & Weather Analysis</p>
</div>
""", unsafe_allow_html=True)

# CLASS NAMES

class_names = [
    "Apple Scab",
    "Apple Black Rot",
    "Corn Rust",
    "Healthy Leaf",
    "Potato Early Blight",
    "Tomato Leaf Mold",
    "Tomato Yellow Leaf Curl Virus"
]

# DISEASE DATABASE

solutions = {

    "Apple Scab": {
        "plant": "Apple Plant",
        "medicine": "Copper Fungicide Spray",
        "cause": "Fungal infection",
        "symptoms": "Dark spots on leaves",
        "chance": "Rainy season",
        "treatment": "Spray every 7 days",
        "organic": "Neem Oil Spray",
        "prevention": "Remove infected leaves"
    },

    "Apple Black Rot": {
        "plant": "Apple Plant",
        "medicine": "Mancozeb Spray",
        "cause": "Black fungal infection",
        "symptoms": "Black spots on leaves",
        "chance": "High humidity",
        "treatment": "Clean infected area",
        "organic": "Baking Soda Spray",
        "prevention": "Avoid overwatering"
    },

    "Corn Rust": {
        "plant": "Corn Plant",
        "medicine": "Azoxystrobin",
        "cause": "Fungal spores",
        "symptoms": "Rust colored spots",
        "chance": "Warm climate",
        "treatment": "Apply fungicide",
        "organic": "Neem Extract",
        "prevention": "Use resistant seeds"
    },

    "Potato Early Blight": {
        "plant": "Potato Plant",
        "medicine": "Chlorothalonil",
        "cause": "Wet fungal disease",
        "symptoms": "Brown dry spots",
        "chance": "Rainy weather",
        "treatment": "Weekly fungicide spray",
        "organic": "Cow urine spray",
        "prevention": "Maintain spacing"
    },

    "Tomato Leaf Mold": {
        "plant": "Tomato Plant",
        "medicine": "Copper Fungicide",
        "cause": "Poor airflow",
        "symptoms": "Yellow spots",
        "chance": "High humidity",
        "treatment": "Reduce humidity",
        "organic": "Neem Oil",
        "prevention": "Increase airflow"
    },

    "Tomato Yellow Leaf Curl Virus": {
        "plant": "Tomato Plant",
        "medicine": "Neem Oil Spray",
        "cause": "Whitefly attack",
        "symptoms": "Leaf curling",
        "chance": "Summer season",
        "treatment": "Control insects",
        "organic": "Garlic Spray",
        "prevention": "Use insect nets"
    },

    "Healthy Leaf": {
        "plant": "Healthy Plant",
        "medicine": "No medicine needed",
        "cause": "Healthy leaf",
        "symptoms": "Fresh green leaves",
        "chance": "No disease",
        "treatment": "Continue proper care",
        "organic": "Organic Compost",
        "prevention": "Maintain watering"
    }
}

# MODEL

@st.cache_resource
def load_model():

    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(class_names), activation="softmax")
    ])

    return model

model = load_model()

# WEATHER API

st.markdown("## 🌦 Weather Analysis")

API_KEY = "YOUR_OPENWEATHER_API_KEY"

city = st.text_input(
    "Enter City Name",
    "Prayagraj"
)

temperature = None
humidity = None
weather = None

try:

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)

    data_weather = response.json()

    if data_weather["cod"] == 200:

        temperature = data_weather["main"]["temp"]

        humidity = data_weather["main"]["humidity"]

        weather = data_weather["weather"][0]["main"]

        col1,col2,col3 = st.columns(3)

        with col1:
            st.metric(
                "🌡 Temperature",
                f"{temperature} °C"
            )

        with col2:
            st.metric(
                "💧 Humidity",
                f"{humidity}%"
            )

        with col3:
            st.metric(
                "☁ Weather",
                weather
            )

except:
    st.warning("⚠ Weather API not connected")


# PREDICTION FUNCTION


def predict_disease(image):

    img = image.convert("RGB")

    img = img.resize((224,224))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = np.max(prediction) * 100

    return predicted_class, confidence

# =========================================================
# IMAGE UPLOAD
# =========================================================

st.markdown("## 📤 Upload Plant Leaf")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg","jpeg","png"]
)

# =========================================================
# IMAGE DETECTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="🌿 Uploaded Leaf",
        width=300
    )

    with st.spinner(
        "🔍 Detecting Disease..."
    ):

        result, confidence = predict_disease(
            image
        )

    data = solutions[result]

    st.success("✅ Detection Completed")

    st.error(f"🦠 Disease: {result}")

    st.info(
        f"🎯 Confidence: {confidence:.2f}%"
    )

    st.markdown("## 🌱 Plant Information")

    st.write(
        f"🌿 Plant Name: {data['plant']}"
    )

    st.write(
        f"🧪 Cause: {data['cause']}"
    )

    st.write(
        f"📌 Symptoms: {data['symptoms']}"
    )

    st.write(
        f"⚠ Disease Chance: {data['chance']}"
    )

    # =====================================================
    # WEATHER ALERT
    # =====================================================

    st.markdown("## 🌦 Weather Disease Alert")

    if humidity and humidity > 70:

        st.warning(
            "⚠ High Humidity Disease Risk"
        )

        st.write("• Leaf Mold")

        st.write("• Rust")

        st.write("💊 Spray Copper Fungicide")

    if weather == "Rain":

        st.error(
            "🌧 Rain Expected"
        )

        st.write(
            "⚠ Fungal disease risk high"
        )

        st.write(
            "💊 Use Mancozeb Spray"
        )

    if temperature and temperature > 35:

        st.warning(
            "🔥 Heat Stress Risk"
        )

        st.write(
            "💧 Increase watering"
        )

    # =====================================================
    # MEDICINE
    # =====================================================

    st.markdown("## 💊 Treatment")

    st.success(
        f"💊 Medicine: {data['medicine']}"
    )

    st.write(
        f"⚡ Treatment: {data['treatment']}"
    )

    st.write(
        f"🌿 Organic: {data['organic']}"
    )

    st.write(
        f"🛡 Prevention: {data['prevention']}"
    )

    # =====================================================
    # HEALTHY LEAF
    # =====================================================

    if "Healthy" in result:

        st.balloons()

        st.success(
            "🌱 Healthy Plant"
        )

    else:

        st.warning(
            "⚠ Immediate Care Needed"
        )

# =========================================================
# CAMERA DETECTION
# =========================================================

st.markdown("---")

st.markdown("## 📷 Live Camera Detection")

st.info("👉 Allow Camera Permission")

camera_image = st.camera_input(
    "Capture Plant Leaf"
)

if camera_image is not None:

    cam_image = Image.open(
        camera_image
    )

    st.image(
        cam_image,
        caption="📷 Camera Image",
        width=300
    )

    with st.spinner(
        "🔍 Detecting Disease..."
    ):

        result, confidence = predict_disease(
            cam_image
        )

    data = solutions[result]

    st.success(
        "✅ Camera Detection Completed"
    )

    st.error(
        f"🦠 Disease: {result}"
    )

    st.info(
        f"🎯 Confidence: {confidence:.2f}%"
    )

    st.write(
        f"🌿 Plant Name: {data['plant']}"
    )

    st.write(
        f"💊 Medicine: {data['medicine']}"
    )

    st.write(
        f"⚡ Treatment: {data['treatment']}"
    )

    st.write(
        f"🛡 Prevention: {data['prevention']}"
    )

    if "Healthy" in result:

        st.balloons()

        st.success(
            "🌱 Healthy Plant"
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<center>

🌾 Smart Sustainable Agriculture AI System

<br><br>

🤖 Plant Disease Detection • Camera Detection • Weather Analysis

</center>
""", unsafe_allow_html=True)


# =====================================================
# 🚜 SMART FARM ROUTE MANAGEMENT
# =====================================================

st.markdown("## 🛣 Smart Farm Routes")

route_name = st.selectbox(
    "Select Farm Route",
    [
        "Route 1 - Apple Farm",
        "Route 2 - Corn Farm",
        "Route 3 - Potato Farm",
        "Route 4 - Tomato Farm"
    ]
)

route_data = {

    "Route 1 - Apple Farm": {
        "crop": "Apple",
        "disease": "Apple Scab, Apple Black Rot",
        "water": "Medium",
        "fertilizer": "Organic Compost + NPK",
        "spray": "Copper Fungicide",
        "season": "Rainy Season"
    },

    "Route 2 - Corn Farm": {
        "crop": "Corn",
        "disease": "Corn Rust",
        "water": "Medium",
        "fertilizer": "Nitrogen Fertilizer",
        "spray": "Azoxystrobin",
        "season": "Summer"
    },

    "Route 3 - Potato Farm": {
        "crop": "Potato",
        "disease": "Potato Early Blight",
        "water": "High",
        "fertilizer": "Potash",
        "spray": "Chlorothalonil",
        "season": "Rainy"
    },

    "Route 4 - Tomato Farm": {
        "crop": "Tomato",
        "disease": "Leaf Mold, Yellow Leaf Curl Virus",
        "water": "Medium",
        "fertilizer": "NPK 19:19:19",
        "spray": "Neem Oil Spray",
        "season": "Summer"
    }
}

selected = route_data[route_name]

st.success(f"✅ Selected : {route_name}")

col1, col2 = st.columns(2)

with col1:

    st.info(f"🌱 Crop : {selected['crop']}")

    st.info(f"🦠 Disease : {selected['disease']}")

    st.info(f"💧 Water Need : {selected['water']}")

with col2:

    st.info(f"🌿 Fertilizer : {selected['fertilizer']}")

    st.info(f"💊 Recommended Spray : {selected['spray']}")

    st.info(f"🌦 Best Season : {selected['season']}")

# =====================================================
# ROUTE WEATHER ADVISORY
# =====================================================

st.markdown("### 🌦 Route Advisory")

if weather == "Rain":

    st.error("🌧 Rain Expected")

    st.write("💊 Spray Before Rain")

    st.write("• Mancozeb")

    st.write("• Copper Fungicide")

    st.write("• Neem Oil")

elif temperature and temperature > 35:

    st.warning("🔥 High Temperature")

    st.write("💧 Water Morning & Evening")

    st.write("🌾 Use Mulching")

elif humidity and humidity > 75:

    st.warning("💧 High Humidity")

    st.write("💊 Copper Fungicide Recommended")

else:

    st.success("✅ Weather Conditions Normal")

# =====================================================
# ROUTE WATER REQUIREMENT
# =====================================================

st.markdown("### 🚿 Route Irrigation")

if temperature and temperature > 35:

    st.error("HIGH WATER REQUIREMENT")

elif humidity and humidity < 40:

    st.warning("MEDIUM WATER REQUIREMENT")

else:

    st.success("NORMAL WATER REQUIREMENT")

# =====================================================
# ROUTE SPRAY RECOMMENDATION
# =====================================================

st.markdown("### 💊 Spray Recommendation")

st.success(selected["spray"])

# =====================================================
# ROUTE FERTILIZER RECOMMENDATION
# =====================================================

st.markdown("### 🌿 Fertilizer Recommendation")

st.success(selected["fertilizer"])
