---
title: Conversion Rate Predictor
emoji: 📧
colorFrom: gray
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# 📧 Conversion Rate Prediction - Data Science Weekly
[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Live%20App-yellow)]
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Container-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
## 🎯 Live Demo
https://huggingface.co/spaces/msshaqura/conversion_rate_prediction_project

## 📊 Project Overview
This application predicts whether a website visitor will subscribe to the **Data Science Weekly** newsletter based on their behavior.

### Key Metrics
| Metric | Value |
|--------|-------|
| **F1-Score** | 0.776 |
| **ROC-AUC** | 0.987 |
| **Precision** | 0.87 |
| **Recall** | 0.69 |

## 🔑 Key Insights from the Model
- Users visiting **11+ pages** have **45.9% conversion rate**
- **Existing users** convert **5x better** than new users
- **Germany** (6.24%) and **UK** (5.25%) show highest conversion rates
- **Total pages visited** is the most important predictor

## 🛠️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/conversion-rate-prediction.git

# Navigate to project folder
cd conversion-rate-prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

📁 Project Structure

├── app.py                          # Streamlit application
├── conversion_model.pkl            # Trained Logistic Regression model
├── preprocessor.pkl                # Data preprocessor (scaler + encoder)
├── feature_info.pkl                # Feature names and threshold
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
└── README.md                       # This file

🤖 Model Details
Algorithm: Logistic Regression

Features: 5 original + 3 encoded = 8 features

Optimal Threshold: 0.45 (tuned for F1-score)

Training Data: 284,580 users

No overfitting: Train F1 (0.762) ≈ Validation F1 (0.767)

📈 Feature Impact (Coefficients)
| Feature	        |Impact	        |Direction |
|---------------    |---------------|--------- |
| Germany	        | +3.57	        | Positive |
| United Kingdom    | +3.40	        | Positive |
| United States	    | +3.06	        | Positive |
| Pages Visited     | +2.53	        | Positive |
| New User	        | -0.79	        | Negative |
| Age	            | -0.60	        | Negative |

💡 Business Recommendations
Increase Page Engagement (Highest Impact)

Add related articles and internal links

Expected impact: +15-20% conversion

Target Existing Users (Quick Win)

Personalized email campaigns

Expected impact: +5-10% conversion

Geographic Targeting

Focus marketing on Germany and UK

Expected impact: +3-5% conversion

📦 Dependencies
Python 3.10+

Streamlit 1.28.0

Scikit-learn 1.4.2

Pandas, NumPy, Matplotlib, Plotly, Seaborn

👥 Team
Conversion Rate Prediction Challenge - Jedha Bootcamp

📅 Date
April 2026

📄 License
MIT