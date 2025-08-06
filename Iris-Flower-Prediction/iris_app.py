import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="🌸 Iris Flower Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin: 10px 0;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Input styling */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 10px;
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("IrisFlower.pkl", "rb"))
    except FileNotFoundError:
        st.error("❌ Model file 'IrisFlower.pkl' not found. Please ensure the file is in the same directory.")
        st.stop()

model = load_model()

# Load sample data for visualization
@st.cache_data
def load_iris_data():
    # Sample Iris dataset for visualization
    return pd.DataFrame({
        'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 7.0, 6.4, 6.9, 5.5, 6.5, 6.3, 6.6, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8],
        'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.2, 3.2, 3.1, 2.3, 2.8, 3.3, 2.9, 3.0, 2.5, 2.9, 2.5, 3.6, 3.0, 2.7, 2.8],
        'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 4.7, 4.5, 4.9, 4.0, 4.6, 6.0, 4.6, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 4.8],
        'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3, 1.5, 2.5, 1.3, 2.1, 1.7, 1.8, 1.8, 2.5, 2.0, 1.9, 1.4],
        'species': ['Setosa', 'Setosa', 'Setosa', 'Setosa', 'Setosa', 'Versicolor', 'Versicolor', 'Versicolor', 'Versicolor', 'Versicolor',
                   'Virginica', 'Virginica', 'Virginica', 'Virginica', 'Virginica', 'Virginica', 'Virginica', 'Virginica', 'Virginica', 'Virginica']
    })

iris_data = load_iris_data()

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None
if 'input_values' not in st.session_state:
    st.session_state.input_values = {'sepal_length': 5.0, 'sepal_width': 3.0, 'petal_length': 4.0, 'petal_width': 1.0}

# Header
st.markdown('<h1 class="main-header">🌸 Iris Flower Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: white; font-size: 1.2rem;">Predict the species of Iris flowers using machine learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Input Parameters")
    st.markdown("Adjust the measurements of your Iris flower:")

    # Synchronized input widgets
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sepal Length (cm)**")
        sepal_length_slider = st.slider("", 4.0, 8.0, st.session_state.input_values['sepal_length'], 0.1, key="sl_slider")
        sepal_length = st.number_input("", 4.0, 8.0, sepal_length_slider, 0.1, key="sl_number")

        st.markdown("**Sepal Width (cm)**")
        sepal_width_slider = st.slider("", 2.0, 5.0, st.session_state.input_values['sepal_width'], 0.1, key="sw_slider")
        sepal_width = st.number_input("", 2.0, 5.0, sepal_width_slider, 0.1, key="sw_number")

    with col2:
        st.markdown("**Petal Length (cm)**")
        petal_length_slider = st.slider("", 1.0, 7.0, st.session_state.input_values['petal_length'], 0.1, key="pl_slider")
        petal_length = st.number_input("", 1.0, 7.0, petal_length_slider, 0.1, key="pl_number")

        st.markdown("**Petal Width (cm)**")
        petal_width_slider = st.slider("", 0.1, 3.0, st.session_state.input_values['petal_width'], 0.1, key="pw_slider")
        petal_width = st.number_input("", 0.1, 3.0, petal_width_slider, 0.1, key="pw_number")

    # Sync sliders and number inputs
    if sepal_length != sepal_length_slider:
        st.session_state.sl_slider = sepal_length
    if sepal_width != sepal_width_slider:
        st.session_state.sw_slider = sepal_width
    if petal_length != petal_length_slider:
        st.session_state.pl_slider = petal_length
    if petal_width != petal_width_slider:
        st.session_state.pw_slider = petal_width

    # Store values
    st.session_state.input_values = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }

    # Predict button
    if st.button("🔮 Predict Species", type="primary", use_container_width=True):
        # Make prediction
        input_df = pd.DataFrame({
            "sepal length (cm)": [sepal_length],
            "sepal width (cm)": [sepal_width],
            "petal length (cm)": [petal_length],
            "petal width (cm)": [petal_width]
        })

        prediction = model.predict(input_df)[0]
        species_names = ['Setosa', 'Versicolor', 'Virginica']

        # Get probabilities if available
        try:
            probabilities = model.predict_proba(input_df)[0]
            st.session_state.probabilities = probabilities
        except:
            st.session_state.probabilities = None

        st.session_state.prediction = species_names[prediction]

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Prediction result
    if st.session_state.prediction:
        species_emojis = {'Setosa': '🌺', 'Versicolor': '🌸', 'Virginica': '🌼'}
        emoji = species_emojis.get(st.session_state.prediction, '🌸')

        st.markdown(f'<div class="prediction-result">{emoji} Predicted Species: {st.session_state.prediction}</div>', 
                   unsafe_allow_html=True)

        # Probability metrics
        if st.session_state.probabilities is not None:
            st.markdown("### 📊 Prediction Confidence")
            prob_col1, prob_col2, prob_col3 = st.columns(3)

            species_names = ['Setosa', 'Versicolor', 'Virginica']
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

            for i, (col, species, prob, color) in enumerate(zip([prob_col1, prob_col2, prob_col3], 
                                                              species_names, 
                                                              st.session_state.probabilities,
                                                              colors)):
                with col:
                    is_predicted = species == st.session_state.prediction
                    icon = "✅" if is_predicted else ""
                    st.metric(
                        label=f"{species} {icon}",
                        value=f"{prob:.1%}",
                        delta=None
                    )

    # Scatter plot visualization
    if st.session_state.prediction:
        st.markdown("### 🎯 Your Flower vs Dataset")

        # Create scatter plot
        fig = px.scatter_matrix(
            iris_data,
            dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            color='species',
            title="Iris Dataset Scatter Matrix",
            color_discrete_map={'Setosa': '#ff6b6b', 'Versicolor': '#4ecdc4', 'Virginica': '#45b7d1'}
        )

        # Add user input as red star
        for i in range(4):
            for j in range(4):
                if i != j:
                    fig.add_trace(go.Scatter(
                        x=[list(st.session_state.input_values.values())[j]],
                        y=[list(st.session_state.input_values.values())[i]],
                        mode='markers',
                        marker=dict(color='red', size=15, symbol='star'),
                        name='Your Input',
                        showlegend=(i == 0 and j == 1)
                    ), row=i+1, col=j+1)

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Information panels
    st.markdown("### 📚 About Iris Species")

    with st.expander("🌺 Iris Setosa", expanded=False):
        st.write("""
        **Characteristics:**
        - Smallest petals
        - Widest sepals
        - Most compact flower
        - Native to Alaska and northeastern US
        """)

    with st.expander("🌸 Iris Versicolor", expanded=False):
        st.write("""
        **Characteristics:**
        - Medium-sized features
        - Moderate petal length
        - Found in eastern North America
        - Also called Blue Flag Iris
        """)

    with st.expander("🌼 Iris Virginica", expanded=False):
        st.write("""
        **Characteristics:**
        - Largest petals and sepals
        - Longest measurements overall
        - Native to eastern United States
        - Also called Southern Blue Flag
        """)

    # How it works
    with st.expander("🤖 How It Works", expanded=False):
        st.write("""
        This app uses a machine learning model trained on the famous Iris dataset:

        1. **Input**: Four flower measurements
        2. **Processing**: ML algorithm analyzes patterns
        3. **Output**: Species prediction with confidence

        The model was trained on 150 samples with 96%+ accuracy!
        """)

    # Current measurements display
    if st.session_state.input_values:
        st.markdown("### 📏 Current Measurements")
        measurements_df = pd.DataFrame({
            'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Value (cm)': [f"{v:.1f}" for v in st.session_state.input_values.values()]
        })
        st.dataframe(measurements_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: white; opacity: 0.7;">🌸 Built with Streamlit | Enhanced Interactive ML Predictor</p>',
    unsafe_allow_html=True
)
