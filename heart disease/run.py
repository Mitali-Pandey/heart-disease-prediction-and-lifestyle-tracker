"""
Quick Start Script
Run this to start the application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import xgboost
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️ NLTK download warning: {e}")
        print("You can download manually later")

def main():
    """Main function"""
    print("🫀 Heart Disease Prediction System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Run Streamlit
    print("\n🚀 Starting Streamlit application...")
    print("The app will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped. Goodbye!")

if __name__ == "__main__":
    main()

