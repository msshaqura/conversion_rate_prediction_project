# app.py
# Conversion Rate Prediction - Data Science Weekly
# Multi-page Streamlit App with STAR methodology

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Conversion Rate Predictor",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for both light and dark mode with hover effects
st.markdown("""
<style>
    /* Responsive design for all screen modes */
    .stApp {
        background-color: var(--background-color);
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
        color: var(--text-color);
    }
    
    .sub-header {
        font-size: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #7F7F7F;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: var(--text-color);
    }
    
    .card {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .metric-highlight {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .metric-highlight:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .metric-highlight h3 {
        font-size: 2rem;
        margin: 0;
    }
    
    .metric-highlight p {
        margin: 0;
        font-size: 0.9rem;
    }
    
    .prediction-box {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid var(--border-color);
    }
    
    .prediction-high {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00AA00;
    }
    
    .prediction-low {
        font-size: 1.3rem;
        color: #FF6B6B;
    }
    
    .stButton button {
        background-color: #000000;
        color: #FFFFFF;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #424242;
        color: #FFFFFF;
        transform: scale(1.05);
    }
    
    hr {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Set matplotlib style for dark/light mode compatibility
plt.style.use('default')
sns.set_style("whitegrid")

# Load model and preprocessor
@st.cache_resource
def load_models():
    """Load trained model and preprocessor"""
    model = joblib.load('conversion_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    feature_info = joblib.load('feature_info.pkl')
    threshold = feature_info['threshold']
    return model, preprocessor, threshold

# Load data for EDA
@st.cache_data
def load_data():
    """Load training data for EDA"""
    df = pd.read_csv('conversion_data_train.csv')
    return df

import streamlit as st

# Sidebar
with st.sidebar:
    
    # Logo 
    st.image("logo_DSC.png", width=200)

    # Title
    st.markdown("""
    <h1 style='color:#0071CE;'>Conversion Rate Prediction Dashboard</h1>
    """, unsafe_allow_html=True)

    # Navigation
    st.header("📧 Navigation")
    page = st.radio(
        "Go to:",
        ["📋 1. Project Overview (STAR)", 
         "🔧 2. Methodology & Data", 
         "📊 3. Exploratory Data Analysis",
         "🎯 4. Live Prediction",
         "💡 5. Conclusions & Recommendations"]
    )

    # Separator
    st.markdown("---")

    # Author section
    st.markdown("### 👨‍💻 Présenté par")
    st.markdown("**Mohammed SHAQURA**")
    st.markdown("Data Analyst | Conversion Rate Prediction Project")
    st.markdown("**Jedha Bootcamp**")

    # Separator
    st.markdown("---")

    # Model Performance
    st.markdown("**Model Performance**")
    st.metric("F1-Score", "0.776", delta="+0.0087")
    st.metric("ROC-AUC", "0.987")
    st.metric("Precision", "0.87")
    st.metric("Recall", "0.69")

# ============================================
# PAGE 1: PROJECT OVERVIEW (STAR METHODOLOGY)
# ============================================
if page == "📋 1. Project Overview (STAR)":
    st.markdown('<div class="main-header">📧 Data Science Weekly</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Conversion Rate Prediction Challenge</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">📌 Situation</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        Data Science Weekly is a popular newsletter curated by independent data scientists.
        The team wants to understand user behavior on their website and identify what drives 
        newsletter subscriptions.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">🎯 Task</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        Build a machine learning model that predicts whether a user will subscribe to the newsletter
        using only behavioral data. The model must achieve a high F1-score due to imbalanced data
        (only ~3% of users convert).
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-title">⚙️ Action</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Complete EDA with visualizations<br>
        • Feature engineering and preprocessing<br>
        • Logistic Regression with threshold tuning<br>
        • Model evaluation using F1-score<br>
        • Analysis of feature importance
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">🏆 Result</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • Best F1-Score: <strong>0.776</strong><br>
        • ROC-AUC: <strong>0.987</strong><br>
        • Key insight: Users visiting 11+ pages convert at <strong>45.9%</strong><br>
        • Existing users convert <strong>5x better</strong> than new users
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key metrics
    st.markdown('<div class="section-title">📊 Key Business Metrics</div>', unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown("""
        <div class="metric-highlight">
        <h3>3.23%</h3>
        <p>Current Conversion Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown("""
        <div class="metric-highlight">
        <h3>45.9%</h3>
        <p>Conversion at 11+ pages</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown("""
        <div class="metric-highlight">
        <h3>7.19%</h3>
        <p>Existing Users Conversion</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown("""
        <div class="metric-highlight">
        <h3>6.24%</h3>
        <p>Germany Conversion</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# PAGE 2: METHODOLOGY & DATA
# ============================================
elif page == "🔧 2. Methodology & Data":
    st.markdown('<div class="main-header">🔧 Methodology & Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">How we built the prediction model</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">📁 Dataset Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <strong>Training Set:</strong> 284,580 users with labels<br>
        <strong>Test Set:</strong> 31,620 users without labels<br>
        <strong>Features:</strong> 5 variables<br>
        <strong>Target:</strong> converted (0/1)<br>
        <strong>Imbalance:</strong> 3.23% positive class
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">📋 Features Description</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        • <strong>country</strong> - User's country (China, UK, Germany, US)<br>
        • <strong>age</strong> - User's age (17-123 years)<br>
        • <strong>new_user</strong> - 1 if new, 0 if existing<br>
        • <strong>source</strong> - Traffic source (Seo, Ads, Direct)<br>
        • <strong>total_pages_visited</strong> - Pages in session (1-29)
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-title">🔄 Preprocessing Steps</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <strong>1. Age cleaning:</strong> Capped outliers at 100 years<br>
        <strong>2. Missing values:</strong> None found<br>
        <strong>3. Encoding:</strong> OneHotEncoder for country and source<br>
        <strong>4. Scaling:</strong> StandardScaler for numerical features<br>
        <strong>5. Split:</strong> 80/20 train/validation with stratification
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">🤖 Model Selection</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <strong>Final Model:</strong> Logistic Regression<br>
        <strong>Why?</strong><br>
        • Simple, interpretable, fast<br>
        • No overfitting (train ≈ validation)<br>
        • Outperformed Random Forest (0.745)<br>
        <strong>Optimal Threshold:</strong> 0.45 (tuned for F1-score)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
        # Pipeline visualization - Professional Flowchart
    st.markdown('<div class="section-title">📊 Modeling Pipeline</div>', unsafe_allow_html=True)

    # Create pipeline steps with details
    pipeline_steps = [
        {"step": "1. Raw Data", "icon": "📁", "details": "284,580 users\n5 features"},
        {"step": "2. Preprocessing", "icon": "🔧", "details": "Scale numerical\nEncode categorical"},
        {"step": "3. Train/Val Split", "icon": "✂️", "details": "80% Train\n20% Validation"},
        {"step": "4. Logistic Regression", "icon": "🤖", "details": "C=1.0\nmax_iter=1000"},
        {"step": "5. Predictions", "icon": "🎯", "details": "Threshold=0.45\nF1=0.776"}
    ]

    # Display pipeline as horizontal cards with arrows
    cols = st.columns(5)

    for i, (col, step_info) in enumerate(zip(cols, pipeline_steps)):
        with col:
            # Card with hover effect
            st.markdown(f"""
            <div style="
                text-align: center;
                background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
                padding: 1rem 0.5rem;
                border-radius: 12px;
                border: 1px solid var(--border-color);
                transition: all 0.3s ease;
                cursor: pointer;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            "
            onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 10px 20px rgba(0,0,0,0.15)';"
            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 5px rgba(0,0,0,0.1)';">
                <div style="font-size: 2rem;">{step_info['icon']}</div>
                <div style="font-weight: bold; font-size: 0.9rem; margin-top: 0.5rem;">{step_info['step']}</div>
                <div style="font-size: 0.7rem; color: #7F7F7F; margin-top: 0.5rem; white-space: pre-line;">{step_info['details']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add arrow between cards (except after last)
        if i < 4:
            with cols[i]:
                st.markdown("""
                <div style="text-align: center; font-size: 2rem; color: #7F7F7F; margin-top: 2rem;">
                →
                </div>
                """, unsafe_allow_html=True)

    # Additional pipeline metrics
    st.markdown("---")
    st.markdown('<div class="section-title">⚙️ Pipeline Details</div>', unsafe_allow_html=True)

    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)

    with col_metrics1:
        st.markdown("""
        <div class="metric-highlight">
        <h3>5</h3>
        <p>Raw Features</p>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics2:
        st.markdown("""
        <div class="metric-highlight">
        <h3>8</h3>
        <p>After Preprocessing</p>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics3:
        st.markdown("""
        <div class="metric-highlight">
        <h3>227,664</h3>
        <p>Training Samples</p>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics4:
        st.markdown("""
        <div class="metric-highlight">
        <h3>56,916</h3>
        <p>Validation Samples</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# PAGE 3: EXPLORATORY DATA ANALYSIS
# ============================================
elif page == "📊 3. Exploratory Data Analysis":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding user behavior patterns</div>', unsafe_allow_html=True)
    
    df = load_data()
    
    st.markdown("---")
    
    # Target variable - CORRECTED with matplotlib
    st.markdown('<div class="section-title">🎯 Target Variable: Conversion</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Using matplotlib for pie chart
        conv_counts = df['converted'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_pie = ['#BDBDBD', '#000000']
        labels_pie = [f'Not Converted\n({conv_counts[0]:,} users, {conv_counts[0]/len(df)*100:.1f}%)', 
                      f'Converted\n({conv_counts[1]:,} users, {conv_counts[1]/len(df)*100:.1f}%)']
        ax.pie(conv_counts.values, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax.set_title('Conversion Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown(f"""
        <div class="card">
        <strong>Key Findings:</strong><br><br>
        • <strong>{conv_counts[0]:,}</strong> users did NOT convert ({conv_counts[0]/len(df)*100:.2f}%)<br>
        • <strong>{conv_counts[1]:,}</strong> users converted ({conv_counts[1]/len(df)*100:.2f}%)<br>
        • This is a <strong>highly imbalanced</strong> dataset<br>
        • F1-score is the right metric (not accuracy)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Numerical features - CORRECTED with matplotlib
    st.markdown('<div class="section-title">📈 Numerical Features Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution with matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['age'], bins=50, color='#000000', alpha=0.7, edgecolor='gray')
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Number of Users', fontsize=12)
        ax.set_title('Age Distribution', fontsize=14, fontweight='bold')
        ax.axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['age'].mean():.1f}")
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        <div class="card">
        <strong>Age Insights:</strong><br>
        • Mean age: 30.6 years<br>
        • Median age: 30 years<br>
        • Most users: 24-36 years old<br>
        • Few outliers above 100 years
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Pages distribution with matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['total_pages_visited'], bins=29, color='#424242', alpha=0.7, edgecolor='gray')
        ax.set_xlabel('Number of Pages', fontsize=12)
        ax.set_ylabel('Number of Users', fontsize=12)
        ax.set_title('Pages Visited Distribution', fontsize=14, fontweight='bold')
        ax.axvline(df['total_pages_visited'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['total_pages_visited'].mean():.1f}")
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        <div class="card">
        <strong>Pages Insights:</strong><br>
        • Mean pages: 4.9<br>
        • Median pages: 4<br>
        • Range: 1-29 pages<br>
        • Right-skewed distribution
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Conversion rate by category - using matplotlib
    st.markdown('<div class="section-title">🌍 Conversion Rate by Category</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        country_conv = df.groupby('country')['converted'].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(country_conv.index, country_conv.values, color='#000000')
        ax.set_xlabel('Conversion Rate (%)', fontsize=10)
        ax.set_title('By Country', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        source_conv = df.groupby('source')['converted'].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(source_conv.index, source_conv.values, color='#424242')
        ax.set_xlabel('Conversion Rate (%)', fontsize=10)
        ax.set_title('By Source', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()
    
    with col3:
        newuser_conv = df.groupby('new_user')['converted'].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(['Existing (0)', 'New (1)'], newuser_conv.values, color=['#000000', '#BDBDBD'])
        ax.set_ylabel('Conversion Rate (%)', fontsize=10)
        ax.set_title('By User Type', fontsize=12, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Most important insight - with matplotlib
    st.markdown('<div class="section-title">⭐ Most Important Insight: Pages vs Conversion</div>', unsafe_allow_html=True)
    
    df_copy = df.copy()
    df_copy['pages_group'] = pd.cut(df_copy['total_pages_visited'], bins=[0, 2, 4, 7, 10, 30], 
                                    labels=['1-2', '3-4', '5-7', '8-10', '11+'])
    pages_conv = df_copy.groupby('pages_group', observed=True)['converted'].mean() * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(pages_conv.index, pages_conv.values, color=['#BDBDBD', '#BDBDBD', '#BDBDBD', '#BDBDBD', '#000000'])
    ax.set_xlabel('Pages Group', fontsize=12)
    ax.set_ylabel('Conversion Rate (%)', fontsize=12)
    ax.set_title('Conversion Rate by Pages Visited', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, pages_conv.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{value:.1f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("""
    <div class="card" style="background-color:#000000; color:#FFFFFF; text-align:center;">
    <strong>🔑 KEY INSIGHT:</strong> Users who visit <strong>11+ pages</strong> have a <strong>45.9% conversion rate</strong>, 
    compared to only 0.5% for users who visit fewer pages. This is the strongest predictor of conversion by far!
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE 4: LIVE PREDICTION
# ============================================
elif page == "🎯 4. Live Prediction":
    st.markdown('<div class="main-header">🎯 Live Conversion Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter user data to predict newsletter subscription</div>', unsafe_allow_html=True)
    
    # Load models
    model, preprocessor, threshold = load_models()
    
    st.markdown("---")
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        country = st.selectbox(
            "🌍 Country",
            options=["China", "UK", "Germany", "US"],
            help="User's country of origin"
        )
    
    with col2:
        age = st.number_input(
            "🎂 Age",
            min_value=17,
            max_value=100,
            value=30,
            help="User's age (17-100 years)"
        )
    
    with col3:
        new_user = st.selectbox(
            "👤 User Type",
            options=[0, 1],
            format_func=lambda x: "Existing User" if x == 0 else "New User",
            help="0 = Existing user, 1 = New user"
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        source = st.selectbox(
            "📢 Traffic Source",
            options=["Direct", "Ads", "Seo"],
            help="How the user arrived at the website"
        )
    
    with col5:
        total_pages = st.slider(
            "📄 Pages Visited",
            min_value=1,
            max_value=30,
            value=5,
            help="Number of pages visited during the session"
        )
    
    with col6:
        st.markdown("### ")
        predict_button = st.button("🔮 Predict Conversion", use_container_width=True)
    
    # Prediction function
    def predict_conversion(country, age, new_user, source, total_pages, model, preprocessor):
        input_data = pd.DataFrame({
            'country': [country],
            'age': [age],
            'new_user': [new_user],
            'source': [source],
            'total_pages_visited': [total_pages]
        })
        
        input_data['age_clean'] = input_data['age'].apply(lambda x: 100 if x > 100 else x)
        feature_columns = ['country', 'age_clean', 'new_user', 'source', 'total_pages_visited']
        X_input = input_data[feature_columns]
        X_preprocessed = preprocessor.transform(X_input)
        prob = model.predict_proba(X_preprocessed)[0][1]
        return prob
    
    # Prediction logic
    if predict_button:
        with st.spinner("Analyzing user behavior..."):
            prob = predict_conversion(country, age, new_user, source, total_pages, model, preprocessor)
            prediction = 1 if prob >= threshold else 0
            
            st.markdown("---")
            
            col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
            
            with col_result2:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown(f'<div class="prediction-high">🎉 LIKELY TO SUBSCRIBE 🎉</div>', unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:1.2rem;'>Probability: <strong>{prob*100:.1f}%</strong></p>", unsafe_allow_html=True)
                    st.markdown("<p>This user has a high probability of subscribing to the newsletter.</p>")
                    st.markdown("<p style='color:#7F7F7F;'>✅ Recommended action: Send a welcome email immediately.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-low">📧 UNLIKELY TO SUBSCRIBE</div>', unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:1.2rem;'>Probability: <strong>{prob*100:.1f}%</strong></p>", unsafe_allow_html=True)
                    st.markdown("<p>This user is unlikely to subscribe at this moment.</p>")
                    st.markdown("<p style='color:#7F7F7F;'>💡 Recommended action: Encourage more page browsing or retarget later.</p>", unsafe_allow_html=True)
                
                # Probability gauge with plotly (works well)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    delta={'reference': threshold * 100, 'relative': True},
                    title={'text': "Conversion Probability (%)"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#000000"},
                        'steps': [
                            {'range': [0, threshold * 100], 'color': "#F0F0F0"},
                            {'range': [threshold * 100, 100], 'color': "#BDBDBD"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PAGE 5: CONCLUSIONS & RECOMMENDATIONS
# ============================================
elif page == "💡 5. Conclusions & Recommendations":
    st.markdown('<div class="main-header">💡 Conclusions & Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Actionable insights for Data Science Weekly</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model summary
    st.markdown('<div class="section-title">🤖 Model Performance Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("F1-Score", "0.776", delta="+0.0087 vs baseline")
    with col2:
        st.metric("ROC-AUC", "0.987", delta="Excellent")
    with col3:
        st.metric("Precision", "0.87", delta="Converted class")
    with col4:
        st.metric("Recall", "0.69", delta="Converted class")
    
    st.markdown("---")
    
    # Feature importance - using matplotlib
    st.markdown('<div class="section-title">📊 Feature Impact (Coefficients)</div>', unsafe_allow_html=True)
    
    feature_importance = pd.DataFrame({
        'Feature': ['Germany', 'United Kingdom', 'United States', 'Pages Visited', 
                    'Age', 'New User', 'Direct Traffic', 'SEO Traffic'],
        'Impact': [3.57, 3.40, 3.06, 2.53, -0.60, -0.79, -0.21, -0.04],
        'Direction': ['Positive', 'Positive', 'Positive', 'Positive', 'Negative', 'Negative', 'Negative', 'Negative']
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_feat = ['green' if x > 0 else 'red' for x in feature_importance['Impact']]
    bars = ax.barh(feature_importance['Feature'], feature_importance['Impact'], color=colors_feat)
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_title('Feature Impact on Conversion (Logistic Regression Coefficients)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add value labels
    for bar, value in zip(bars, feature_importance['Impact']):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, f'{value:.2f}', 
                va='center', fontsize=10)
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Top 5 Recommendations
    st.markdown('<div class="section-title">📈 Top 5 Recommendations</div>', unsafe_allow_html=True)
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("""
        <div class="card">
        <h3>🥇 1. Increase Page Engagement</h3>
        <p>Users visiting 11+ pages have <strong>45.9% conversion rate</strong>.<br>
        <strong>Actions:</strong> Add related articles, internal links, reading progress bars.<br>
        <strong>Expected impact:</strong> +15-20% conversion</p>
        </div>
        
        <div class="card">
        <h3>🥈 2. Target Existing Users</h3>
        <p>Existing users convert at <strong>7.19%</strong> vs new users at 1.41% (5x better).<br>
        <strong>Actions:</strong> Personalized emails, loyalty program, re-engagement campaigns.<br>
        <strong>Expected impact:</strong> +5-10% conversion</p>
        </div>
        
        <div class="card">
        <h3>🥉 3. Geographic Targeting</h3>
        <p>Germany (6.24%) and UK (5.25%) outperform other countries.<br>
        <strong>Actions:</strong> Increase marketing spend in Germany/UK, investigate low conversion in China.<br>
        <strong>Expected impact:</strong> +3-5% conversion</p>
        </div>
        """, unsafe_allow_html=True)
    
    with rec_col2:
        st.markdown("""
        <div class="card">
        <h3>4. Age Demographic Focus</h3>
        <p>18-25 age group converts at <strong>5.13%</strong> (best among all ages).<br>
        <strong>Actions:</strong> Create content tailored to young adults, use social media channels.<br>
        <strong>Expected impact:</strong> +2-3% conversion</p>
        </div>
        
        <div class="card">
        <h3>5. Optimize Existing Channels</h3>
        <p>All sources perform similarly (Ads 3.48%, SEO 3.29%, Direct 2.78%).<br>
        <strong>Actions:</strong> Maintain current channel mix, slight increase in Ad spend.<br>
        <strong>Expected impact:</strong> +1-2% conversion</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Priority matrix - CORRECTED with matplotlib
    st.markdown('<div class="section-title">🎯 Priority Action Matrix</div>', unsafe_allow_html=True)
    
    priority_data = pd.DataFrame({
        'Recommendation': ['Page Engagement', 'Target Existing Users', 'Geographic Targeting', 'Age Focus', 'Channel Optimization'],
        'Impact': [5, 4, 3, 2, 1],
        'Difficulty': [3, 1, 2, 1, 1],
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_priority = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC']
    scatter = ax.scatter(priority_data['Difficulty'], priority_data['Impact'], 
                         s=500, c=colors_priority, alpha=0.7)
    
    # Add labels
    for i, row in priority_data.iterrows():
        ax.annotate(row['Recommendation'], 
                   (row['Difficulty'], row['Impact']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Implementation Difficulty (1=Easy, 5=Hard)', fontsize=12)
    ax.set_ylabel('Business Impact (1=Low, 5=High)', fontsize=12)
    ax.set_title('Priority Action Matrix', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add quadrants
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax.text(4.2, 4.5, 'High Impact\nHard to Implement', fontsize=9, alpha=0.5, ha='center')
    ax.text(1.2, 4.5, 'High Impact\nEasy to Implement', fontsize=9, alpha=0.5, ha='center')
    ax.text(4.2, 1.5, 'Low Impact\nHard to Implement', fontsize=9, alpha=0.5, ha='center')
    ax.text(1.2, 1.5, 'Low Impact\nEasy to Implement', fontsize=9, alpha=0.5, ha='center')
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Quick win
    st.markdown("""
    <div class="card" style="background-color:#000000; color:#FFFFFF; text-align:center;">
    <h3>⚡ QUICK WIN - Implement First</h3>
    <p>Add a <strong>"Recommended for you"</strong> section after the 5th page to keep users browsing longer.<br>
    Low development effort, high potential return on investment.</p>
    </div>
    """, unsafe_allow_html=True)
    
# ============================================
# FOOTER 
# ============================================
# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'> Conversion Rate Prediction Dashboard | Created by Streamlit </p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'> Project developed as part of the BLOC 3 certification | Jedha Bootcamp </p>",
    unsafe_allow_html=True
)