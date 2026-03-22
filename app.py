# ============================================
# TITANIC SURVIVAL PREDICTION - ADVANCED UI
# ============================================

import streamlit as st
import pickle
import numpy as np

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# -------- LOAD MODEL --------
model = pickle.load(open('model.pkl', 'rb'))

# ============================================
# 🎨 CUSTOM CSS (UI DESIGN)
# ============================================

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

h1 {
    text-align: center;
    color: #FFD700;
}

.stButton>button {
    background-color: #ff4b2b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #ff416c;
    transform: scale(1.05);
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.success {
    background-color: #1abc9c;
}

.danger {
    background-color: #e74c3c;
}
</style>
""", unsafe_allow_html=True)


# ============================================
# 🎬 HEADER SECTION
# ============================================

st.markdown("<h1>🚢 Titanic Survival Prediction</h1>", unsafe_allow_html=True)

st.image(
    "https://images.unsplash.com/photo-1541417904950-b855846fe074",
    use_column_width=True
)

st.markdown("### 💡 Predict whether a passenger would survive the Titanic disaster")


# ============================================
# 📥 INPUT SECTION (2 COLUMNS)
# ============================================

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("🎟 Passenger Class", [1, 2, 3])
    sex = st.selectbox("👤 Gender", ["Male", "Female"])
    age = st.slider("🎂 Age", 0, 100, 25)

with col2:
    sibsp = st.slider("👨‍👩‍👧 Siblings/Spouse", 0, 10, 0)
    parch = st.slider("👶 Parents/Children", 0, 10, 0)
    fare = st.number_input("💰 Fare", 0.0, 500.0, 50.0)

# Embarked
embarked = st.selectbox("📍 Embarked", ["S", "C", "Q"])


# ============================================
# 🔄 ENCODING
# ============================================

sex = 1 if sex == "Male" else 0

if embarked == "S":
    embarked = 2
elif embarked == "C":
    embarked = 0
else:
    embarked = 1


# ============================================
# 🔮 PREDICTION BUTTON
# ============================================

if st.button("🚀 Predict Survival"):

    # Animation spinner
    with st.spinner("Analyzing passenger data... ⏳"):
        
        input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        prediction = model.predict(input_data)

    # ============================================
    # 🎉 RESULT DISPLAY (ANIMATED STYLE)
    # ============================================

    if prediction[0] == 1:
        st.markdown(
            '<div class="result-box success">🎉 Passenger Survived!</div>',
            unsafe_allow_html=True
        )
        st.balloons()   # 🎈 animation

    else:
        st.markdown(
            '<div class="result-box danger">💀 Passenger Did Not Survive</div>',
            unsafe_allow_html=True
        )


# ============================================
# 📊 SIDEBAR
# ============================================

st.sidebar.title("📌 About Project")

st.sidebar.info("""
This AI model predicts Titanic survival using Machine Learning.

👨‍💻 Developed by Sen  
🚀 Tech Stack:
- Python
- Scikit-learn
- Streamlit
""")

st.sidebar.markdown("---")
st.sidebar.write("🔥 Make smarter predictions with AI")