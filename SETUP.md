# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation Steps

1. **Clone or download the project**
   ```bash
   cd "heart disease"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
   ```
   
   Or run:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('vader_lexicon')
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Register a new account or use existing credentials
   - Start using the system!

## First Time Setup

1. **Register an account**: Click "Register" in the sidebar
2. **Train models**: Navigate to "Model Training & Comparison" and click "Train Models"
3. **Make predictions**: Go to "Make Prediction" and enter patient data

## Troubleshooting

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're using Python 3.8+

### NLTK Data Missing
- Run the NLTK download commands above
- Check that nltk_data directory exists in your home directory

### Database Errors
- The database will be created automatically on first run
- If issues persist, delete `heart_disease.db` and restart

### Model Training Issues
- Ensure you have sufficient memory (models can be memory-intensive)
- Training may take several minutes - be patient
- Check console for error messages

## Features Overview

- **Multi-Model Comparison**: Train and compare 5 different ML models
- **Explainable AI**: SHAP values, LIME explanations, feature importance
- **Real-Time Predictions**: Instant risk assessment
- **Risk Calculator**: Framingham-style risk calculation
- **Symptom Analyzer**: NLP-based symptom analysis
- **Lifestyle Recommendations**: Personalized health advice
- **User Authentication**: Secure login system
- **Prediction History**: Track all predictions
- **PDF Reports**: Generate comprehensive reports
- **AI Chatbot**: Interactive health assistant
- **Voice Analysis**: Stress detection from voice

## Notes

- This is an educational/research tool - not for clinical use
- Always consult healthcare professionals for medical decisions
- Sample data is generated if no dataset is provided
- Models are trained on sample data by default

