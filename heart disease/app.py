"""
Main Streamlit Application
Heart Disease Prediction System - Interactive Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from features.data_cleaning import DataCleaner, load_sample_data
from models.model_trainer import ModelTrainer
from features.risk_calculator import RiskCalculator
from features.symptom_analyzer import SymptomAnalyzer
from features.lifestyle_recommender import LifestyleRecommender
# from models.hyperparameter_tuning import HyperparameterTuner  # commented per request
# from models.ensemble_model import EnsembleModel  # commented per request
from utils.database import Database
from utils.auth import Auth
from utils.pdf_generator import PDFReportGenerator
from features.chatbot import HealthChatbot

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c5aa0;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4788;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f4788;
        color: white;
        font-weight: bold;
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Initialize components
auth = Auth()
db = Database()
risk_calc = RiskCalculator()
symptom_analyzer = SymptomAnalyzer()
lifestyle_recommender = LifestyleRecommender()
pdf_generator = PDFReportGenerator()
chatbot = HealthChatbot()

def main():
    """Main application"""
    # Header
    st.markdown('<div class="main-header">🫀 Heart Disease Prediction System</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_main_app()

def show_login_page():
    """Show login/registration page"""
    st.sidebar.title("🔐 Authentication")
    
    tab1, tab2 = st.sidebar.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            user, message = auth.login_user(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user['id']
                st.session_state.username = user['username']
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error(message)
    
    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Username", key="reg_username")
        new_email = st.text_input("Email", key="reg_email")
        new_password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            else:
                user_id, message = auth.register_user(new_username, new_email, new_password)
                if user_id:
                    st.success("Registration successful! Please login.")
                else:
                    st.error(message)
    
    # Show welcome message
    st.markdown("""
    ## Welcome to the Heart Disease Prediction System
    
    This comprehensive ML-based system provides:
    
    - **Multi-Model Comparison**: Compare Logistic Regression, Random Forest, XGBoost, SVM, and Neural Networks
    - **Risk assessment**: ML predictions (after training) and a Framingham-style calculator side by side
    - **Symptom Analysis**: NLP-based symptom analysis
    - **Lifestyle Recommendations**: Personalized health advice
    - **And much more!**
    
    Please login or register to continue.
    """)

def show_main_app():
    """Show main application"""
    # Sidebar
    st.sidebar.title(f"👤 {st.session_state.username}")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()
    
    # Navigation
    nav_options = [
        "🏠 Dashboard",
        "🔮 Risk assessment",
        "📊 Model Training & Comparison",
        "💬 Symptom Analyzer",
        "💡 Lifestyle Recommendations",
        # "🎯 Hyperparameter Tuning",  # commented per request
        # "🔧 Ensemble Model",  # commented per request
        "📁 Prediction History",
        "🤖 AI Health Chatbot",
        "📄 Generate PDF Report",
        "🧹 Data Cleaning Module",
    ]
    page = st.sidebar.selectbox("Navigate", nav_options, key="main_nav")
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔮 Risk assessment":
        show_risk_assessment()
    elif page == "📊 Model Training & Comparison":
        show_model_training()
    elif page == "💬 Symptom Analyzer":
        show_symptom_analyzer()
    elif page == "💡 Lifestyle Recommendations":
        show_lifestyle_recommendations()
    # elif page == "🎯 Hyperparameter Tuning":
    #     show_hyperparameter_tuning()
    # elif page == "🔧 Ensemble Model":
    #     show_ensemble_model()
    elif page == "📁 Prediction History":
        show_prediction_history()
    elif page == "🤖 AI Health Chatbot":
        show_chatbot()
    elif page == "📄 Generate PDF Report":
        show_pdf_generator()
    elif page == "🧹 Data Cleaning Module":
        show_data_cleaning()

def show_dashboard():
    """Show main dashboard"""
    st.title("📊 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get user stats
    predictions = db.get_user_predictions(st.session_state.user_id, limit=100)
    
    with col1:
        st.metric("Total Predictions", len(predictions))
    
    with col2:
        if predictions:
            avg_risk = np.mean([p['risk_score'] for p in predictions if p['risk_score']])
            st.metric("Average Risk Score", f"{avg_risk:.1f}%" if avg_risk else "N/A")
        else:
            st.metric("Average Risk Score", "N/A")
    
    with col3:
        if predictions:
            latest = predictions[0]
            st.metric("Latest Risk", f"{latest['risk_score']:.1f}%" if latest['risk_score'] else "N/A")
        else:
            st.metric("Latest Risk", "N/A")
    
    with col4:
        if predictions:
            high_risk_count = sum(1 for p in predictions if p.get('risk_score', 0) >= 50)
            st.metric("High Risk Predictions", high_risk_count)
        else:
            st.metric("High Risk Predictions", 0)
    
    # Prediction trend chart
    if predictions:
        st.subheader("📈 Prediction History Trend")
        df_history = pd.DataFrame(predictions)
        df_history['created_at'] = pd.to_datetime(df_history['created_at'])
        df_history = df_history.sort_values('created_at')
        
        fig = px.line(df_history, x='created_at', y='risk_score',
                     title='Risk Score Over Time',
                     labels={'risk_score': 'Risk Score (%)', 'created_at': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔮 Open risk assessment"):
            st.session_state["main_nav"] = "🔮 Risk assessment"
            st.rerun()
    
    with col2:
        if st.button("📊 Train Models"):
            st.session_state.page = "Model Training"
            st.rerun()
    
    with col3:
        if st.button("💬 Ask Chatbot"):
            st.session_state.page = "Chatbot"
            st.rerun()

def show_risk_assessment():
    """ML classifier and Framingham-style calculator in one place (different inputs and methods)."""
    st.title("🔮 Risk assessment")
    st.markdown(
        "Use **ML model prediction** after training models on the heart dataset features. "
        "Use **Framingham risk score** when you have BP, lipids, smoking, and diabetes history—no model training required. "
        "Numbers may differ because the methods are not identical."
    )
    tab_ml, tab_fram = st.tabs(["ML model prediction", "Framingham risk score"])

    with tab_ml:
        _show_ml_prediction_panel()

    with tab_fram:
        _show_framingham_risk_panel()


_ML_PREDICTION_TERM_EXPLANATIONS = {
    "— Hide explanations —": "",
    "Age": (
        "**Age** — The person’s age in years. Heart risk generally increases as people get older."
    ),
    "Sex": (
        "**Sex** — Biological sex recorded in the clinic. Some heart-risk patterns differ on average "
        "between male and female patients."
    ),
    "Chest pain type": (
        "**Chest pain type** — This is **not something you guess from the internet**. It is how doctors **classify** your symptom story "
        "(and sometimes tests).\n\n"
        "- **Typical angina** — Chest pressure/tightness brought on by **walking, climbing stairs, or exertion**, often eases within "
        "minutes **when you stop and rest**. Think “predictable with effort, better with rest.”\n"
        "- **Atypical angina** — Chest symptoms that are **unclear or only partly** fit that pattern (e.g. sharp, fleeting, or mainly at rest).\n"
        "- **Non-anginal pain** — Discomfort that **doesn’t sound like classic heart pain** to the clinician (may still need evaluation).\n"
        "- **Asymptomatic** — **No chest-pain symptoms** you reported; the workup was for other reasons.\n\n"
        "**If you are unsure**, choose the closest plain description or ask your clinician which category fits your records."
    ),
    "Resting blood pressure": (
        "**Resting blood pressure** — Pressure in the arteries while at rest, in **mmHg** (same units as a home BP cuff). "
        "Higher values can strain the heart and arteries."
    ),
    "Serum cholesterol": (
        "**Cholesterol (mg/dL)** — A blood fat measure from a lab test. High values are linked to plaque buildup in arteries."
    ),
    "Fasting blood sugar": (
        "**Fasting blood sugar > 120 mg/dL** — Whether blood sugar was high after not eating. "
        "It can reflect diabetes or prediabetes, which affects vessel health."
    ),
    "Resting ECG": (
        "**Resting ECG** — A snapshot of the heart’s electrical pattern at rest. "
        "**ST-T changes** or **left ventricular hypertrophy** can suggest strain or past damage; **normal** is the usual healthy pattern."
    ),
    "Maximum heart rate (stress test)": (
        "**Max heart rate in a stress test** — You **do not calculate this yourself** on a watch during normal day activity. "
        "It is recorded during a **medical exercise stress test** (treadmill or bike with monitoring).\n\n"
        "**Where to find it:** On the printed **stress test report** (often labeled “peak HR”, “maximum heart rate”, or on the time–HR graph). "
        "If you never had a stress test, **leave learning mode** and use the **Framingham** tab instead, or ask your doctor—**do not guess**.\n\n"
        "**Why it varies:** Age, fitness, beta-blockers and other drugs, and how long you could exercise all change the peak number."
    ),
    "Exercise-induced angina": (
        "**Exercise-induced angina** — Whether chest pain or discomfort appears during the stress test. **Yes** can mean "
        "the heart needs more oxygen than the blood supply provides during exertion."
    ),
    "ST depression (exercise)": (
        "**ST depression** — How much a part of the ECG (**ST segment**) dips during exercise, in millimeters. "
        "Larger values can suggest reduced blood flow to the heart muscle during the test."
    ),
    "Slope of ST segment": (
        "**ST segment slope** — This comes from the **ECG during the stress test**, not from how you feel. "
        "The ST segment is part of the heartbeat tracing; its **tilt** at peak exercise is read by a doctor or the lab.\n\n"
        "- **Upsloping** — Segment trends **upward**; often a **milder** pattern but still interpreted by cardiology.\n"
        "- **Flat** — Little or **no upward slope**; can be associated with reduced blood flow during exercise.\n"
        "- **Downsloping** — Segment trends **downward**; often taken **more seriously** and usually needs specialist follow-up.\n\n"
        "**You should use the value written on your stress ECG report** (or what your doctor told you). If you don’t have it, this field isn’t for home guessing."
    ),
    "Major heart vessels (0–3)": (
        "**Major vessels affected (0–3)** — This is from **coronary angiography** (a procedure where dye and X-rays show the heart arteries), "
        "**not** from a routine checkup or home device.\n\n"
        "The number is how many of the **main coronary arteries** show **important narrowing**—your **cardiologist** decides that from the images. "
        "**0** often means none reported as major; **1–3** means one to three vessels involved.\n\n"
        "**If you never had an angiogram**, you should **not** pick a number to “try it”—use the **Framingham** tab for a simpler assessment, "
        "or only use this app with **demo / teaching** values your instructor provides."
    ),
    "Thalassemia / perfusion (thal)": (
        "**Thal (blood flow to heart)** — A coded result from imaging or clinical assessment of blood flow to the heart muscle. "
        "**Normal**, **fixed defect**, and **reversible defect** are categories a cardiologist uses; use whatever value is on the report."
    ),
}


def _show_ml_prediction_panel():
    """Trained classifier on 13 clinical features; saves to prediction history."""
    st.subheader("ML model prediction")
    with st.expander("Who can fill this out? (read this first)", expanded=False):
        st.markdown(
            "This form matches a **clinical cardiology dataset**. Several items only make sense if they come from **real "
            "reports or your doctor**, not from guessing.\n\n"
            "| If you have… | What to do |\n"
            "|--------------|------------|\n"
            "| **No stress test / angiogram** | Use the **Framingham risk score** tab (BP, cholesterol, HDL, smoking, diabetes). "
            "Or use **demo values** only for coursework—not for your own health decisions. |\n"
            "| **Stress test report** | **Peak / max heart rate**, **ST depression**, **ST slope**, and **exercise chest pain** "
            "are on that report or in the doctor’s summary. |\n"
            "| **Angiogram report** | **Major vessels (0–3)** comes from that procedure only. |\n"
            "| **Chest pain type** | Your clinician classifies your **symptom pattern**; use the option that best matches what "
            "your records say, or ask at your appointment. |\n\n"
            "**This screen is not a substitute for care.** Wrong inputs produce meaningless outputs."
        )
    if not st.session_state.models_trained:
        st.warning("⚠️ Models not trained yet. Please train models first from the Model Training & Comparison page.")

    explain_keys = list(_ML_PREDICTION_TERM_EXPLANATIONS.keys())
    picked = st.selectbox(
        "Need help? Pick a field to see what it means",
        explain_keys,
        index=0,
        key="ml_pred_term_explainer",
    )
    if picked != "— Hide explanations —":
        st.markdown(_ML_PREDICTION_TERM_EXPLANATIONS[picked])
        st.caption("Educational only—enter values from a clinician or lab report when possible; this app is not a diagnosis.")

    with st.form("ml_prediction_form"):
        st.markdown("**Patient information** (same features as the training dataset)")
        st.caption("Hover the ⓘ next to each field for a short hint, or use the menu above for a longer explanation.")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=50,
                key="ml_pred_age",
                help="Your age in full years.",
            )
            sex = st.selectbox(
                "Sex",
                [("Male", 1), ("Female", 0)],
                format_func=lambda x: x[0],
                key="ml_pred_sex",
                help="Biological sex as recorded for the patient.",
            )[1]
            cp = st.selectbox(
                "Chest pain type (doctor’s classification)",
                [
                    ("Typical angina — tightness with exertion, better with rest", 0),
                    ("Atypical — partly fits heart pain pattern", 1),
                    ("Non-anginal — unlikely classic angina", 2),
                    ("Asymptomatic — no chest pain reported", 3),
                ],
                format_func=lambda x: x[0],
                key="ml_pred_cp",
                help="How your symptoms were classified clinically—not a self-diagnosis label.",
            )[1]
            trestbps = st.number_input(
                "Resting blood pressure (mmHg)",
                min_value=50,
                max_value=250,
                value=120,
                key="ml_pred_trestbps",
                help="Upper number from a BP reading at rest (systolic), in mmHg.",
            )
            chol = st.number_input(
                "Cholesterol (mg/dL)",
                min_value=100,
                max_value=600,
                value=200,
                key="ml_pred_chol",
                help="Total cholesterol from a blood test.",
            )
            fbs = st.selectbox(
                "Fasting blood sugar over 120 mg/dL?",
                [("No", 0), ("Yes", 1)],
                format_func=lambda x: x[0],
                key="ml_pred_fbs",
                help="Was fasting blood glucose above 120 mg/dL?",
            )[1]

        with col2:
            restecg = st.selectbox(
                "Resting ECG result",
                [
                    ("Normal", 0),
                    ("ST-T Wave Abnormality", 1),
                    ("Left Ventricular Hypertrophy", 2),
                ],
                format_func=lambda x: x[0],
                key="ml_pred_restecg",
                help="Electrical pattern of the heart at rest from an ECG.",
            )[1]
            thalach = st.number_input(
                "Max heart rate in stress test (bpm)",
                min_value=50,
                max_value=220,
                value=150,
                key="ml_pred_thalach",
                help="From the treadmill/bike stress test report (“peak HR”). Not your casual walking HR—if unknown, don’t guess; use Framingham tab.",
            )
            exang = st.selectbox(
                "Chest pain during exercise test?",
                [("No", 0), ("Yes", 1)],
                format_func=lambda x: x[0],
                key="ml_pred_exang",
                help="Did angina (chest discomfort) occur during the test?",
            )[1]
            oldpeak = st.number_input(
                "ST depression (mm)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="ml_pred_oldpeak",
                help="ST segment depression on ECG during exercise, in millimeters.",
            )
            slope = st.selectbox(
                "ST segment slope (stress ECG)",
                [
                    ("Upsloping — ST trends upward at peak exercise", 0),
                    ("Flat — little upward slope", 1),
                    ("Downsloping — ST trends downward", 2),
                ],
                format_func=lambda x: x[0],
                key="ml_pred_slope",
                help="Read from the exercise ECG report; cardiologist interpretation. Not something you feel in the chest.",
            )[1]
            ca = st.number_input(
                "Major vessels narrowed (0–3, angiogram only)",
                min_value=0,
                max_value=3,
                value=0,
                key="ml_pred_ca",
                help="Only if you had coronary angiography. 0 = none major on report. If you never had this test, do not invent a number.",
            )
            thal = st.selectbox(
                "Heart blood flow category (thal)",
                [
                    ("Normal", 0),
                    ("Fixed Defect", 1),
                    ("Reversible Defect", 2),
                ],
                format_func=lambda x: x[0],
                key="ml_pred_thal",
                help="Coded perfusion/imaging result—use the value from the medical report.",
            )[1]
        
        submitted = st.form_submit_button("🔮 Predict Risk")
        
        if submitted and st.session_state.best_model:
            # Prepare features (raw row — scaling lives inside the trained Pipeline / calibrator)
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            
            prediction = st.session_state.best_model.predict(features)[0]
            prediction_proba = st.session_state.best_model.predict_proba(features)[0]
            
            risk_probability = prediction_proba[1]
            
            # Calculate risk score
            risk_assessment = risk_calc.get_risk_category(risk_probability * 100)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "Heart Disease Risk" if prediction == 1 else "No Risk")
            
            with col2:
                st.metric("Probability (calibrated)", f"{risk_probability * 100:.2f}%")
            
            with col3:
                st.metric("Risk Category", risk_assessment['category'])
            
            # Risk assessment
            st.subheader("📊 Risk Assessment")
            risk_color = risk_assessment['color']
            st.markdown(f"""
            <div style="background-color: {risk_color}; padding: 1rem; border-radius: 0.5rem; color: white;">
                <h3>{risk_assessment['category']}</h3>
                <p>{risk_assessment['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Save prediction
            features_dict = {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
                'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
            }
            
            db.save_prediction(
                st.session_state.user_id,
                int(prediction),
                float(risk_probability),
                risk_score=risk_probability * 100,
                risk_category=risk_assessment['category'],
                features=features_dict,
                model_used=st.session_state.best_model_name
            )
            
            st.success("Prediction saved to history!")

def train_models_with_sample_data():
    """Train models with sample data"""
    # Load sample data
    df = load_sample_data()
    
    # Clean data
    cleaner = DataCleaner(df)
    cleaner.remove_duplicates()
    cleaner.handle_missing_values()
    df_clean = cleaner.get_cleaned_data()
    
    # Prepare features
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = df_clean[feature_cols]
    y = df_clean['target']
    
    st.session_state.feature_names = feature_cols
    
    # Train models
    trainer = ModelTrainer(X, y)
    trainer.train_all_models()
    trainer.evaluate_all_models()
    
    # Get best model
    best_name, best_model, best_results = trainer.get_best_model()
    
    st.session_state.models_trained = True
    st.session_state.trained_models = trainer.models
    st.session_state.best_model = best_model
    st.session_state.best_model_name = best_name
    st.session_state.scaler = trainer.scaler
    st.session_state.trainer = trainer
    
    st.success(
        f"✅ Models trained! **{best_name}** selected by highest **5-fold CV F1** on training data "
        f"(not by test-set scores). Probabilities are **sigmoid-calibrated**; test metrics below include **bootstrap 95% CIs**."
    )

def show_model_training():
    """Show model training page"""
    st.title("📊 Model Training & Comparison")
    st.markdown(
        "**Reliability practices in this build:** scaling inside **pipelines** (no leakage in CV), "
        "**model choice** from **5-fold CV F1** on the training split only, **sigmoid calibration** of probabilities, "
        "and **bootstrap 95% intervals** for test-set ROC-AUC and F1 (small samples → wide intervals are normal)."
    )
    
    if st.button("🔄 Train All Models"):
        with st.spinner("Training models... This may take a few minutes."):
            train_models_with_sample_data()
    
    if st.session_state.models_trained and 'trainer' in st.session_state:
        trainer = st.session_state.trainer
        
        # Comparison table
        st.subheader("📈 Model Comparison")
        st.caption(
            "Table sorted by **CV F1 (select)** — the metric used to pick the default model. "
            "Test columns are **held-out** and shown with uncertainty bands."
        )
        comparison_df = trainer.get_comparison_dataframe()
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curves")
            fig_roc = trainer.plot_roc_curves()
            st.pyplot(fig_roc)
        
        with col2:
            st.subheader("Confusion Matrices")
            fig_cm = trainer.plot_confusion_matrices()
            st.pyplot(fig_cm)
        
        # Best model info
        st.subheader("🏆 Best Model (CV-selected)")
        if st.session_state.best_model_name in trainer.results:
            best_results = trainer.results[st.session_state.best_model_name]
            st.info(
                f"**{st.session_state.best_model_name}** — chosen by highest mean **F1** in **5-fold stratified CV** "
                f"on the training split (**{best_results['cv_f1_mean']:.3f} ± {best_results['cv_f1_std']:.3f}**). "
                f"Held-out test metrics below are for **reporting only**."
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Test accuracy", f"{best_results['accuracy']:.3f}")
            col2.metric("Test precision", f"{best_results['precision']:.3f}")
            col3.metric("Test recall", f"{best_results['recall']:.3f}")
            col4.metric("Test F1", f"{best_results['f1_score']:.3f}")
            st.markdown(
                f"**Test ROC-AUC:** {best_results['roc_auc']:.3f} "
                f"_(95% bootstrap CI {best_results['auc_ci_low']:.3f}–{best_results['auc_ci_high']:.3f})_  \n"
                f"**Test F1:** {best_results['f1_score']:.3f} "
                f"_(95% bootstrap CI {best_results['f1_ci_low']:.3f}–{best_results['f1_ci_high']:.3f})_"
            )
        else:
            st.warning("Best model results not found — try training again.")

def _show_framingham_risk_panel():
    """Framingham-style score; does not use trained ML models."""
    st.subheader("Framingham risk score")
    st.caption("Uses blood pressure, lipids, smoking, diabetes, and fasting glucose—no model training required.")
    
    with st.form("framingham_risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50, key="fram_age")
            sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0], key="fram_sex")[1]
            bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120, key="fram_bp_sys")
            bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80, key="fram_bp_dia")
        
        with col2:
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, key="fram_chol")
            hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50, key="fram_hdl")
            fasting_glucose = st.number_input(
                "Fasting Glucose (mg/dL)",
                min_value=50,
                max_value=400,
                value=100,
                key="fram_fasting_glucose",
                help="Optional glycemic input. 100-125 suggests prediabetes; >=126 suggests diabetes range."
            )
            smoking = st.selectbox("Smoking", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="fram_smoke")[1]
            diabetes = st.selectbox("Diabetes", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="fram_diab")[1]
        
        submitted = st.form_submit_button("Calculate risk")
        
        if submitted:
            risk_score, risk_percentage = risk_calc.calculate_framingham_risk(
                age, sex, bp_systolic, bp_diastolic, cholesterol, hdl, smoking, diabetes, fasting_glucose
            )
            
            risk_assessment = risk_calc.get_risk_category(risk_percentage)
            
            st.success("Risk calculated!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{risk_score:.0f}")
            col2.metric("Risk Percentage", f"{risk_percentage:.1f}%")
            col3.metric("Category", risk_assessment['category'])
            
            st.markdown(f"""
            <div style="background-color: {risk_assessment['color']}; padding: 1rem; border-radius: 0.5rem; color: white;">
                <h3>{risk_assessment['category']}</h3>
                <p>{risk_assessment['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk factors breakdown
            st.subheader("Risk Factors Breakdown")
            factors = risk_calc.get_risk_factors_breakdown(
                age, sex, bp_systolic, bp_diastolic, cholesterol, hdl, smoking, diabetes, fasting_glucose
            )
            if factors:
                st.dataframe(pd.DataFrame(factors), use_container_width=True)

def show_symptom_analyzer():
    """Show symptom analyzer page"""
    st.title("💬 Symptom Analyzer")
    st.caption("Select symptoms using checkboxes. You can also add optional notes.")

    symptom_options = [
        ("Chest pain/discomfort", "chest pain"),
        ("Shortness of breath", "shortness of breath"),
        ("Palpitations/irregular heartbeat", "palpitations"),
        ("Fatigue/weakness", "fatigue"),
        ("Dizziness/lightheadedness", "dizziness"),
        ("Sweating/cold sweat", "cold sweat"),
        ("Nausea/vomiting", "nausea"),
        ("Arm/shoulder pain", "left arm pain"),
        ("Jaw pain", "jaw pain"),
        ("Upper back pain", "upper back pain"),
    ]

    selected_symptoms = []
    severity_levels = ["mild", "moderate", "severe"]
    with st.form("symptom_checkbox_form"):
        st.subheader("Select observed symptoms")
        col1, col2 = st.columns(2)
        half = (len(symptom_options) + 1) // 2

        with col1:
            for label, token in symptom_options[:half]:
                if st.checkbox(label, key=f"symptom_chk_{token.replace(' ', '_')}"):
                    severity = st.selectbox(
                        f"Intensity for {label}",
                        severity_levels,
                        index=1,
                        key=f"symptom_sev_{token.replace(' ', '_')}"
                    )
                    selected_symptoms.append(f"{severity} {token}")
        with col2:
            for label, token in symptom_options[half:]:
                if st.checkbox(label, key=f"symptom_chk_{token.replace(' ', '_')}"):
                    severity = st.selectbox(
                        f"Intensity for {label}",
                        severity_levels,
                        index=1,
                        key=f"symptom_sev_{token.replace(' ', '_')}"
                    )
                    selected_symptoms.append(f"{severity} {token}")

        additional_notes = st.text_area(
            "Optional additional notes",
            placeholder="Example: severe since morning, gets worse while climbing stairs..."
        )
        submitted = st.form_submit_button("Analyze Symptoms")

    if submitted:
        if not selected_symptoms and not additional_notes.strip():
            st.warning("Please select at least one symptom or provide additional notes.")
            return

        symptom_text = ", ".join(selected_symptoms)
        if additional_notes.strip():
            symptom_text = f"{symptom_text}. {additional_notes.strip()}" if symptom_text else additional_notes.strip()

        analysis = symptom_analyzer.analyze_symptoms(symptom_text)

        st.subheader("Analysis Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Score", f"{analysis['risk_score']:.1f}")
        with col2:
            st.metric("Risk Category", analysis['risk_category'])

        st.subheader("Symptoms Detected")
        st.write(f"Found {analysis['symptom_count']} symptom(s)")
        for symptom in analysis['symptoms_found']:
            st.write(f"- **{symptom['category']}**: {symptom['keyword']} (Severity: {symptom['severity']})")

        st.subheader("Urgency Assessment")
        urgency = analysis['urgency']
        urgency_color = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}[urgency['level']]
        st.markdown(f"""
        <div style="background-color: {urgency_color}; padding: 1rem; border-radius: 0.5rem; color: white;">
            <h3>{urgency['level']} Urgency</h3>
            <p><strong>Timeframe:</strong> {urgency['timeframe']}</p>
            <p><strong>Recommendation:</strong> {urgency['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Possible Causes")
        for cause in analysis['possible_causes']:
            st.write(f"**{cause['condition']}** ({cause['probability']} probability)")
            st.write(cause['description'])

        st.subheader("Recommendations")
        for rec in analysis['recommendations']:
            st.write(f"• {rec}")

def show_lifestyle_recommendations():
    """Show lifestyle recommendations page"""
    st.title("💡 Lifestyle Recommendations")
    st.caption("Build your profile and get a personalized heart-health action plan.")

    with st.form("lifestyle_form"):
        st.markdown("### Your Health Profile")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Personal & Vitals")
            age = st.number_input("Age", min_value=1, max_value=120, value=50, help="Age in years")
            sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
            bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120)
            bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)

        with col2:
            st.markdown("#### Labs & Lifestyle")
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
            hdl = st.number_input("HDL (mg/dL)", min_value=20, max_value=100, value=50)
            smoking = st.selectbox("Smoking", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            diabetes = st.selectbox("Diabetes", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            physical_activity = st.number_input("Physical Activity (min/week)", min_value=0, max_value=1000, value=150)
            sleep_hours = st.number_input("Sleep Hours/Night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            stress_level = st.slider("Stress Level (0-10)", min_value=0, max_value=10, value=5)

        st.markdown("---")
        submitted = st.form_submit_button("✨ Generate Personalized Plan", use_container_width=True)

    if submitted:
        recommendations = lifestyle_recommender.analyze_user_profile(
            age, sex, bmi, bp_systolic, bp_diastolic, cholesterol, hdl,
            smoking, diabetes, physical_activity, sleep_hours, stress_level
        )

        st.success("Your lifestyle plan is ready.")

        # Quick snapshot
        st.markdown("### Quick Health Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Blood Pressure", f"{bp_systolic}/{bp_diastolic}")
        c2.metric("BMI", f"{bmi:.1f}")
        c3.metric("Activity / Week", f"{physical_activity} min")
        c4.metric("Sleep", f"{sleep_hours:.1f} h/night")

        high_priority_count = 0
        critical_count = 0
        for items in recommendations.values():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        if item.get("priority") == "High":
                            high_priority_count += 1
                        elif item.get("priority") == "Critical":
                            critical_count += 1

        st.info(
            f"Priority focus: {critical_count} critical area(s), {high_priority_count} high-priority area(s). "
            "Start with those first for maximum benefit."
        )

        st.markdown("### Personalized Recommendations")
        tabs = st.tabs(["🥗 Diet", "🏃 Exercise", "😴 Sleep", "🧘 Stress", "🩺 Monitoring", "🧭 General"])
        section_keys = ["diet", "exercise", "sleep", "stress", "monitoring", "general"]

        priority_colors = {
            "Critical": "#b91c1c",
            "High": "#dc2626",
            "Medium": "#d97706",
            "Low": "#15803d",
        }

        for tab, section_key in zip(tabs, section_keys):
            with tab:
                section_items = recommendations.get(section_key, [])
                if not section_items:
                    st.caption("No specific recommendations in this section.")
                    continue

                for item in section_items:
                    if not isinstance(item, dict):
                        continue

                    priority = item.get("priority", "Medium")
                    color = priority_colors.get(priority, "#475569")
                    category = item.get("category", section_key.title())

                    st.markdown(
                        f"""
                        <div style="border-left: 6px solid {color}; background-color: #f8fafc; padding: 0.8rem 1rem; border-radius: 0.5rem; margin: 0.6rem 0;">
                            <strong>{category}</strong> &nbsp; <span style="color:{color}; font-weight:600;">{priority} Priority</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if "recommendations" in item:
                        for rec in item.get("recommendations", []):
                            st.write(f"- {rec}")

                    if "weekly_goal" in item:
                        st.caption(f"Weekly Goal: {item['weekly_goal']}")
                    if "target_hours" in item:
                        st.caption(f"Target Sleep: {item['target_hours']} hours/night")

                    monitoring_plan = item.get("monitoring_plan", [])
                    if monitoring_plan:
                        st.markdown("**Monitoring Plan**")
                        st.dataframe(pd.DataFrame(monitoring_plan), use_container_width=True)


def show_prediction_history():
    """Show prediction history"""
    st.title("📁 Prediction History")
    
    predictions = db.get_user_predictions(st.session_state.user_id)
    
    if predictions:
        df = pd.DataFrame(predictions)
        st.dataframe(df, use_container_width=True)
        
        # Statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(df))
        if df['risk_score'].notna().any():
            col2.metric("Average Risk", f"{df['risk_score'].mean():.1f}%")
            col3.metric("Latest Risk", f"{df.iloc[0]['risk_score']:.1f}%" if df.iloc[0]['risk_score'] else "N/A")
    else:
        st.info("No predictions yet. Make your first prediction!")

def show_chatbot():
    """Show chatbot page"""
    st.title("🤖 AI Health Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about heart health, predictions, or recommendations..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get chatbot response
        response = chatbot.process_query(prompt)
        
        # Add assistant response
        assistant_message = response.get('response', 'I apologize, I did not understand that.')
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        with st.chat_message("assistant"):
            st.markdown(assistant_message)
            
            if 'suggestions' in response:
                st.write("**Suggestions:**")
                for suggestion in response['suggestions']:
                    st.write(f"• {suggestion}")

def show_pdf_generator():
    """Show PDF generator page"""
    st.title("📄 Generate PDF Report")
    
    st.info("Generate a comprehensive PDF report of your latest prediction.")
    
    if st.button("Generate Report"):
        predictions = db.get_user_predictions(st.session_state.user_id, limit=1)
        
        if predictions:
            latest = predictions[0]
            features = latest.get('features', {})
            
            # Generate PDF
            pdf_path = f"report_{st.session_state.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            pdf_generator.generate_report(
                pdf_path,
                features,
                {
                    'prediction': latest['prediction_result'],
                    'probability': latest['prediction_probability']
                },
                {
                    'category': latest.get('risk_category', 'Unknown'),
                    'risk_percentage': latest.get('risk_score', 0),
                    'recommendation': "Consult with healthcare provider"
                },
                {},
                {'model_name': latest.get('model_used', 'Unknown')}
            )
            
            st.success("PDF report generated!")
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("Download PDF", pdf_file, file_name=pdf_path)
        else:
            st.warning("No predictions found. Make a prediction first!")

def show_data_cleaning():
    """Show data cleaning module"""
    st.title("🧹 Data Cleaning Module")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Original Data")
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape}")
        
        cleaner = DataCleaner(df)
        
        # Duplicates
        duplicates = cleaner.detect_duplicates()
        st.metric("Duplicates Found", duplicates)
        
        if st.button("Remove Duplicates"):
            df_clean = cleaner.remove_duplicates()
            st.success("Duplicates removed!")
        
        # Missing values
        missing_df = cleaner.detect_missing_values()
        if not missing_df.empty:
            st.subheader("Missing Values")
            st.dataframe(missing_df)
        
        # Outliers
        if st.button("Detect Outliers (Z-score)"):
            outliers = cleaner.detect_outliers_zscore()
            st.json(outliers)
        
        # Correlation
        if st.button("Show Correlation Heatmap"):
            fig = cleaner.plot_correlation_heatmap()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
