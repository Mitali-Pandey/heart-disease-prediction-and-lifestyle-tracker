# 🫀 Heart Disease Prediction System - Advanced ML Project

A comprehensive machine learning system for heart disease prediction with explainable AI, real-time dashboard, and advanced features.

## 🌟 Features

### Core ML Features
- **Multi-Model Comparison**: Compare Logistic Regression, Random Forest, XGBoost, SVM, and Neural Networks
- **Explainable AI**: SHAP values, LIME explanations, and feature importance visualization
- **Real-Time Prediction Dashboard**: Interactive Streamlit interface

### Advanced Features
- **Risk Score Calculator**: Framingham-style risk assessment
- **Symptom-Based NLP Analysis**: Natural language symptom processing
- **Lifestyle Recommendations**: Personalized health suggestions
- **Data Cleaning Module**: Outlier detection and data quality checks
- **Hyperparameter Tuning**: GridSearchCV, RandomizedSearch, and Optuna
- **Ensemble Model**: Combined model predictions

### User Experience
- **User Authentication**: Login system with prediction history
- **PDF Report Generation**: Doctor-friendly reports
- **Voice Stress Analysis**: Emotion detection from voice
- **AI Health Chatbot**: Interactive health assistant

## 🚀 Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

4. Run the application:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
heart-disease-prediction/
├── app.py                      # Main Streamlit application
├── models/
│   ├── model_trainer.py       # Model training and comparison
│   ├── ensemble_model.py      # Ensemble model implementation
│   └── hyperparameter_tuning.py
├── xai/
│   └── explainability.py      # SHAP, LIME, feature importance
├── features/
│   ├── data_cleaning.py       # Data preprocessing
│   ├── risk_calculator.py     # Risk score calculator
│   ├── symptom_analyzer.py    # NLP symptom analysis
│   ├── lifestyle_recommender.py
│   ├── voice_analysis.py      # Voice stress detection
│   └── chatbot.py             # AI health chatbot
├── utils/
│   ├── database.py            # User database management
│   ├── pdf_generator.py       # PDF report generation
│   └── auth.py                # Authentication system
├── data/
│   └── heart_disease.csv      # Dataset (to be added)
└── saved_models/              # Trained models storage
```

## 🎯 Usage

1. **Start the application**: `streamlit run app.py`
2. **Login or Register**: Create an account or login
3. **Enter Patient Data**: Fill in the prediction form
4. **View Predictions**: See model comparisons and explanations
5. **Get Recommendations**: Receive personalized lifestyle advice
6. **Generate Reports**: Download PDF reports

## ⚠️ Medical Disclaimer

This tool is for educational and research purposes only. It does not replace professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 📊 Model Performance

The system compares multiple models and automatically selects the best performing one based on accuracy, precision, recall, and F1-score.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is for educational purposes.

