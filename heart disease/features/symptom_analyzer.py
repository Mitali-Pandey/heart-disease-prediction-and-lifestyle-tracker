"""
Symptom-Based Text Analysis Module
Uses NLP to analyze natural language symptom descriptions
"""

import re
import numpy as np
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class SymptomAnalyzer:
    """Analyze symptoms from natural language text"""
    
    def __init__(self):
        self.heart_related_keywords = {
            'chest': ['chest', 'chest pain', 'chest discomfort', 'pressure', 'tightness'],
            'breathing': ['shortness of breath', 'breathlessness', 'difficulty breathing', 'dyspnea'],
            'pain': ['pain', 'ache', 'discomfort', 'soreness'],
            'heart': ['heart', 'cardiac', 'palpitations', 'irregular heartbeat', 'arrhythmia'],
            'fatigue': ['fatigue', 'tiredness', 'exhaustion', 'weakness'],
            'dizziness': ['dizziness', 'lightheaded', 'faint', 'vertigo'],
            'sweating': ['sweating', 'perspiration', 'cold sweat'],
            'nausea': ['nausea', 'nauseous', 'vomiting'],
            'arm': ['arm pain', 'left arm', 'right arm', 'shoulder'],
            'jaw': ['jaw pain', 'jaw discomfort'],
            'back': ['back pain', 'upper back']
        }
        
        self.urgency_keywords = {
            'high': ['severe', 'intense', 'crushing', 'sudden', 'acute', 'emergency', 'extreme'],
            'medium': ['moderate', 'mild', 'occasional', 'sometimes'],
            'low': ['slight', 'minor', 'barely', 'rarely']
        }
        
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Preprocess input text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def extract_symptoms(self, text):
        """Extract heart-related symptoms from text"""
        text_lower = text.lower()
        found_symptoms = []
        
        for category, keywords in self.heart_related_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_symptoms.append({
                        'category': category,
                        'keyword': keyword,
                        'severity': self._assess_severity(text_lower, keyword)
                    })
                    break  # Only count each category once
        
        return found_symptoms
    
    def _assess_severity(self, text, keyword):
        """Assess severity based on surrounding words"""
        # Find keyword position
        idx = text.find(keyword)
        if idx == -1:
            return 'medium'
        
        # Check surrounding context
        context_start = max(0, idx - 50)
        context_end = min(len(text), idx + len(keyword) + 50)
        context = text[context_start:context_end]
        
        # Check for urgency keywords
        for urgency_level, urgency_words in self.urgency_keywords.items():
            for word in urgency_words:
                if word in context:
                    return urgency_level
        
        return 'medium'
    
    def calculate_risk_score(self, symptoms):
        """Calculate risk score based on symptoms"""
        if not symptoms:
            return 0
        
        risk_score = 0
        symptom_weights = {
            'chest': 30,
            'breathing': 25,
            'heart': 20,
            'pain': 15,
            'fatigue': 10,
            'dizziness': 10,
            'sweating': 10,
            'nausea': 8,
            'arm': 5,
            'jaw': 5,
            'back': 5
        }
        
        severity_multipliers = {
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5
        }
        
        for symptom in symptoms:
            category = symptom['category']
            severity = symptom['severity']
            
            base_score = symptom_weights.get(category, 5)
            multiplier = severity_multipliers.get(severity, 1.0)
            risk_score += base_score * multiplier
        
        # Normalize to 0-100
        risk_score = min(risk_score, 100)
        
        return risk_score
    
    def assess_urgency(self, text, symptoms, risk_score):
        """Assess urgency level"""
        text_lower = text.lower()
        
        # Check for high urgency indicators
        high_urgency_phrases = [
            'emergency', 'call 911', 'ambulance', 'severe chest pain',
            'can\'t breathe', 'crushing', 'sudden onset'
        ]
        
        for phrase in high_urgency_phrases:
            if phrase in text_lower:
                return {
                    'level': 'High',
                    'score': 9,
                    'recommendation': 'Seek immediate medical attention. Call emergency services if symptoms worsen.',
                    'timeframe': 'Immediate'
                }
        
        # Check risk score
        if risk_score >= 70:
            return {
                'level': 'High',
                'score': 8,
                'recommendation': 'Consult a cardiologist as soon as possible. Monitor symptoms closely.',
                'timeframe': 'Within 24 hours'
            }
        elif risk_score >= 40:
            return {
                'level': 'Medium',
                'score': 5,
                'recommendation': 'Schedule an appointment with your doctor. Keep track of symptoms.',
                'timeframe': 'Within a week'
            }
        else:
            return {
                'level': 'Low',
                'score': 3,
                'recommendation': 'Monitor symptoms. Consider lifestyle modifications.',
                'timeframe': 'Routine check-up'
            }
    
    def get_possible_causes(self, symptoms):
        """Suggest possible heart-related causes"""
        causes = []
        
        symptom_categories = [s['category'] for s in symptoms]
        
        if 'chest' in symptom_categories and 'breathing' in symptom_categories:
            causes.append({
                'condition': 'Angina or Coronary Artery Disease',
                'probability': 'High',
                'description': 'Chest pain with breathing difficulty may indicate reduced blood flow to heart'
            })
        
        if 'chest' in symptom_categories and 'arm' in symptom_categories:
            causes.append({
                'condition': 'Possible Heart Attack',
                'probability': 'High',
                'description': 'Chest pain radiating to arm requires immediate medical evaluation'
            })
        
        if 'heart' in symptom_categories:
            causes.append({
                'condition': 'Arrhythmia or Palpitations',
                'probability': 'Medium',
                'description': 'Irregular heartbeat may indicate electrical conduction issues'
            })
        
        if 'breathing' in symptom_categories and 'fatigue' in symptom_categories:
            causes.append({
                'condition': 'Heart Failure',
                'probability': 'Medium',
                'description': 'Breathing difficulty with fatigue may indicate reduced heart function'
            })
        
        if not causes:
            causes.append({
                'condition': 'General Cardiovascular Concern',
                'probability': 'Low',
                'description': 'Symptoms may be related to cardiovascular health. Further evaluation recommended.'
            })
        
        return causes
    
    def analyze_symptoms(self, text):
        """Comprehensive symptom analysis"""
        if not text or len(text.strip()) == 0:
            return {
                'error': 'No symptoms provided'
            }
        
        # Extract symptoms
        symptoms = self.extract_symptoms(text)
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(symptoms)
        
        # Assess urgency
        urgency = self.assess_urgency(text, symptoms, risk_score)
        
        # Get possible causes
        possible_causes = self.get_possible_causes(symptoms)
        
        # Sentiment analysis
        sentiment_scores = self.sia.polarity_scores(text)
        
        return {
            'input_text': text,
            'symptoms_found': symptoms,
            'symptom_count': len(symptoms),
            'risk_score': risk_score,
            'risk_category': self._categorize_risk(risk_score),
            'urgency': urgency,
            'possible_causes': possible_causes,
            'sentiment': {
                'compound': sentiment_scores['compound'],
                'label': 'Negative' if sentiment_scores['compound'] < -0.1 else 
                        'Positive' if sentiment_scores['compound'] > 0.1 else 'Neutral'
            },
            'recommendations': self._generate_recommendations(symptoms, risk_score)
        }
    
    def _categorize_risk(self, risk_score):
        """Categorize risk level"""
        if risk_score >= 70:
            return 'High Risk'
        elif risk_score >= 40:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    def _generate_recommendations(self, symptoms, risk_score):
        """Generate personalized recommendations"""
        recommendations = []
        
        if risk_score >= 70:
            recommendations.append("Seek immediate medical evaluation")
            recommendations.append("Avoid physical exertion until evaluated")
            recommendations.append("Have someone monitor you")
        
        if 'chest' in [s['category'] for s in symptoms]:
            recommendations.append("Rest and avoid activities that trigger chest discomfort")
            recommendations.append("Monitor blood pressure regularly")
        
        if 'breathing' in [s['category'] for s in symptoms]:
            recommendations.append("Avoid lying flat if breathing is difficult")
            recommendations.append("Consider using extra pillows when sleeping")
        
        if 'heart' in [s['category'] for s in symptoms]:
            recommendations.append("Avoid caffeine and stimulants")
            recommendations.append("Practice stress reduction techniques")
        
        if not recommendations:
            recommendations.append("Maintain regular exercise routine")
            recommendations.append("Follow a heart-healthy diet")
            recommendations.append("Schedule routine cardiovascular screening")
        
        return recommendations

