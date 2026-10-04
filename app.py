import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Olive Oil Variety Predictor",
    page_icon="🫒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful olive oil theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #FFF8DC 0%, #F5E6D3 100%);
    }
    
    /* Title styling */
    h1, h2, h3 {
        color: #556B2F !important;
        font-family: 'Georgia', serif;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #DAA520 0%, #B8860B 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        color: white;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #808000 0%, #6B8E23 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #6B8E23 0%, #556B2F 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F5F5DC 0%, #DEB887 100%);
    }
    
    /* Input fields */
    .stNumberInput>label {
        color: #556B2F;
        font-weight: bold;
    }
    
    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, #FFD700 0%, #DAA520 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        margin: 20px 0;
    }
    
    .prediction-text {
        font-size: 32px;
        font-weight: bold;
        color: #2F4F4F;
        margin: 10px 0;
    }
    
    .confidence-text {
        font-size: 24px;
        color: #556B2F;
        margin: 10px 0;
    }
    
    /* Info boxes */
    .info-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #DAA520;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Warning */
    .warning-box {
        background: #FFF8DC;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF8C00;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_resource
def load_metadata():
    with open("model_meta.json", "r") as f:
        return json.load(f)

model = load_model()
meta = load_metadata()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/olive-oil.png", width=80)
    st.title("🫒 Olive Oil Analyzer")
    st.markdown("---")
    st.markdown("**Variety Prediction System**")
    st.markdown("Predict Tunisian olive oil varieties using chemical composition")
    st.markdown("---")
    
    st.markdown("###  Model Performance")
    st.markdown(f"**Accuracy:** {meta['performance']['accuracy']}")
    st.markdown(f"**Balanced Acc:** {meta['performance']['balanced_accuracy']}")
    st.markdown(f"**F1 Score:** {meta['performance']['f1_score']}")
    st.markdown("---")
    st.markdown("### 📝 About")
    st.markdown("This tool predicts whether an olive oil sample is **Chetoui** (Northern Tunisia) or **Chemlali** (Central/Southern Tunisia) based on its chemical composition.")
    st.markdown("---")
    st.caption("Developed for CBBC - Olive Biotechnology Laboratory")

# Main content
st.title(" Tunisian Olive Oil Variety Predictor")
st.markdown("*Advanced machine learning for olive oil authentication*")
st.markdown("---")

# Create two columns for input sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Fatty Acid Composition")
    st.markdown("*Enter the six major fatty acids (%)*")
    
    c160 = st.number_input("C16:0 (Palmitic acid)", 
                           min_value=0.0, max_value=100.0, 
                           value=float(meta['feature_stats']['c160']['median']),
                           step=0.01, format="%.2f")
    c161 = st.number_input("C16:1 (Palmitoleic acid)", 
                           min_value=0.0, max_value=100.0,
                           value=float(meta['feature_stats']['c161']['median']),
                           step=0.01, format="%.2f")
    c180 = st.number_input("C18:0 (Stearic acid)", 
                           min_value=0.0, max_value=100.0,
                           value=float(meta['feature_stats']['c180']['median']),
                           step=0.01, format="%.2f")
    c181 = st.number_input("C18:1 (Oleic acid)", 
                           min_value=0.0, max_value=100.0,
                           value=float(meta['feature_stats']['c181']['median']),
                           step=0.01, format="%.2f")
    c182 = st.number_input("C18:2 (Linoleic acid)", 
                           min_value=0.0, max_value=100.0,
                           value=float(meta['feature_stats']['c182']['median']),
                           step=0.01, format="%.2f")
    c183 = st.number_input("C18:3 (Linolenic acid)", 
                           min_value=0.0, max_value=100.0,
                           value=float(meta['feature_stats']['c183']['median']),
                           step=0.01, format="%.2f")

with col2:
    st.subheader("🔬 Quality Indices")
    st.markdown("*Optional - leave at median if unknown*")
    
    use_quality = st.checkbox("Include quality indices", value=False)
    
    if use_quality:
        acidite = st.number_input("Free Acidity (%)", 
                                  min_value=0.0, max_value=10.0,
                                  value=float(meta['feature_stats']['acidite']['median']),
                                  step=0.01, format="%.2f")
        peroxyde = st.number_input("Peroxide Value (meq O₂/kg)",
                                   min_value=0.0, max_value=50.0,
                                   value=float(meta['feature_stats']['peroxyde']['median']),
                                   step=0.1, format="%.1f")
        k232 = st.number_input("K232",
                               min_value=0.0, max_value=10.0,
                               value=float(meta['feature_stats']['k232']['median']),
                               step=0.01, format="%.2f")
        k270 = st.number_input("K270",
                               min_value=0.0, max_value=5.0,
                               value=float(meta['feature_stats']['k270']['median']),
                               step=0.01, format="%.2f")
    else:
        acidite = float(meta['feature_stats']['acidite']['median'])
        peroxyde = float(meta['feature_stats']['peroxyde']['median'])
        k232 = float(meta['feature_stats']['k232']['median'])
        k270 = float(meta['feature_stats']['k270']['median'])
        st.info("Quality indices will be set to median values")

# Calculate sum of fatty acids
fa_sum = c160 + c161 + c180 + c181 + c182 + c183

# Validation warnings
warning_col1, warning_col2 = st.columns(2)
with warning_col1:
    if fa_sum > 100:
        st.error(f"⚠️ Fatty acid sum ({fa_sum:.1f}%) exceeds 100%!")
    elif fa_sum < 95:
        st.warning(f"️ Fatty acid sum ({fa_sum:.1f}%) is below 95%")
    else:
        st.success(f"✓ Fatty acid sum: {fa_sum:.1f}%")

# Predict button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🔮 Predict Variety", use_container_width=True)

if predict_btn:
    # Prepare input
    input_data = pd.DataFrame({
        'c160': [c160], 'c161': [c161], 'c180': [c180], 'c181': [c181],
        'c182': [c182], 'c183': [c183], 'acidite': [acidite],
        'peroxyde': [peroxyde], 'k232': [k232], 'k270': [k270]
    })
    
    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    
    # Determine variety
    if prediction == 0:
        variety = "Chetoui"
        confidence = proba[0] * 100
        emoji = "🌲"
        color = "#2F4F4F"
        region = "Northern Tunisia"
    else:
        variety = "Chemlali"
        confidence = proba[1] * 100
        emoji = ""
        color = "#8B4513"
        region = "Central/Southern Tunisia"
    
    # Display prediction
    st.markdown("---")
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-text" style="color: {color}">{emoji} {variety}</div>', 
                unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-text">Confidence: {confidence:.1f}%</div>', 
                unsafe_allow_html=True)
    st.markdown(f'<div style="color: #556B2F; font-size: 18px;">📍 Typical of {region}</div>', 
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Confidence warning
    if confidence < 60:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ **Low confidence prediction** - The sample has characteristics of both varieties. Consider additional analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Probability breakdown
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.metric("Chetoui Probability", f"{proba[0]*100:.1f}%")
    with col_prob2:
        st.metric("Chemlali Probability", f"{proba[1]*100:.1f}%")
    
    # Feature importance
    st.markdown("---")
    st.subheader("📊 Most Important Features")
    
    # Get top 5 features
    importance_df = pd.DataFrame({
        'Feature': list(meta['feature_importance'].keys()),
        'Importance': list(meta['feature_importance'].values())
    }).sort_values('Importance', ascending=False).head(5)
    
    fig = alt.Chart(importance_df).mark_bar(
        cornerRadius=5,
        fill='#DAA520'
    ).encode(
        y=alt.Y('Feature:N', sort='-x', axis=alt.Axis(labelColor='#556B2F', titleColor='#556B2F')),
        x=alt.X('Importance:Q', axis=alt.Axis(title='Importance', labelColor='#556B2F', titleColor='#556B2F')),
        tooltip=['Feature', 'Importance']
    ).properties(
        height=200
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )
    
    st.altair_chart(fig, use_container_width=True)
    
    # Comparison with class profiles
    st.markdown("---")
    st.subheader(" Your Sample vs Average Profiles")
    
    comparison_data = []
    for feat in meta['features'][:6]:  # Only fatty acids
        comparison_data.append({
            'Fatty Acid': feat,
            'Your Sample': input_data[feat].values[0],
            'Chetoui Avg': meta['class_profiles']['Chetoui'][feat],
            'Chemlali Avg': meta['class_profiles']['Chemlali'][feat]
        })
    
    comp_df = pd.DataFrame(comparison_data)
    comp_df_melted = comp_df.melt(id_vars=['Fatty Acid'], 
                                   value_vars=['Your Sample', 'Chetoui Avg', 'Chemlali Avg'],
                                   var_name='Category', value_name='Value')
    
    fig2 = alt.Chart(comp_df_melted).mark_bar(
        cornerRadius=3
    ).encode(
        x=alt.X('Fatty Acid:N', axis=alt.Axis(labelAngle=0, labelColor='#556B2F', titleColor='#556B2F')),
        y=alt.Y('Value:Q', axis=alt.Axis(labelColor='#556B2F', titleColor='#556B2F')),
        color=alt.Color('Category:N', 
                       scale=alt.Scale(range=['#FFD700', '#2F4F4F', '#8B4513'])),
        tooltip=['Fatty Acid', 'Category', 'Value']
    ).properties(
        height=300
    ).configure_axis(
        grid=True,
        gridColor='#F5F5DC'
    )
    
    st.altair_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #8B7355; padding: 20px;">', 
            unsafe_allow_html=True)
st.markdown("🫒 **Olive Oil Variety Prediction System**")
st.markdown("Developed for the Olive Biotechnology Laboratory (LBO) - CBBC")
st.markdown("Academic Year 2025-2026 | Ibtihel AGUIBI")
st.markdown('</div>', unsafe_allow_html=True)