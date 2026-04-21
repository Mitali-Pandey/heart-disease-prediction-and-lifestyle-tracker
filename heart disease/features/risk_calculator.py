"""
Framingham-style Risk Score Calculator
Calculates heart disease risk based on clinical parameters
"""

import numpy as np
import pandas as pd


class RiskCalculator:
    """Calculate heart disease risk score"""
    
    def __init__(self):
        self.risk_factors = {}
        
    def calculate_framingham_risk(self, age, sex, bp_systolic, bp_diastolic,
                                  cholesterol, hdl, smoking, diabetes,
                                  fasting_glucose=None):
        """
        Calculate Framingham-style risk score
        Simplified version for demonstration
        """
        risk_score = 0
        
        # Age factor
        if age < 40:
            risk_score += 0
        elif age < 50:
            risk_score += 5
        elif age < 60:
            risk_score += 10
        elif age < 70:
            risk_score += 15
        else:
            risk_score += 20
        
        # Sex factor (men have higher risk)
        if sex == 1:  # Male
            risk_score += 5
        else:
            risk_score += 2
        
        # Blood pressure factor
        if bp_systolic >= 140 or bp_diastolic >= 90:
            risk_score += 15
        elif bp_systolic >= 130 or bp_diastolic >= 85:
            risk_score += 10
        elif bp_systolic >= 120 or bp_diastolic >= 80:
            risk_score += 5
        
        # Cholesterol factor
        if cholesterol >= 240:
            risk_score += 15
        elif cholesterol >= 200:
            risk_score += 10
        elif cholesterol >= 180:
            risk_score += 5
        
        # HDL factor (lower HDL = higher risk)
        if hdl < 40:
            risk_score += 10
        elif hdl < 50:
            risk_score += 5
        
        # Smoking factor
        if smoking == 1:
            risk_score += 15
        
        # Diabetes / glycemic factor
        if diabetes == 1:
            risk_score += 10
        elif fasting_glucose is not None:
            # Approximate glycemic risk when diabetes status is negative/unknown.
            if fasting_glucose >= 126:
                risk_score += 10
            elif fasting_glucose >= 100:
                risk_score += 5
        
        # Convert score to percentage (0-100%)
        risk_percentage = min(risk_score * 1.5, 100)
        
        return risk_score, risk_percentage
    
    def calculate_ml_based_risk(self, model, features_dict, scaler=None):
        """Calculate risk using ML model prediction"""
        # Convert features dict to array
        feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        feature_array = []
        for feat in feature_order:
            if feat in features_dict:
                feature_array.append(features_dict[feat])
            else:
                feature_array.append(0)  # Default value
        
        feature_array = np.array(feature_array).reshape(1, -1)
        
        # Scale if scaler provided
        if scaler is not None:
            feature_array = scaler.transform(feature_array)
        
        # Get prediction probability
        risk_probability = model.predict_proba(feature_array)[0][1]
        risk_percentage = risk_probability * 100
        
        return risk_percentage
    
    def get_risk_category(self, risk_percentage):
        """Categorize risk level"""
        if risk_percentage < 20:
            category = "Low Risk"
            color = "green"
            recommendation = "Maintain healthy lifestyle. Regular check-ups recommended."
        elif risk_percentage < 50:
            category = "Medium Risk"
            color = "orange"
            recommendation = "Consult with a cardiologist. Consider lifestyle modifications."
        else:
            category = "High Risk"
            color = "red"
            recommendation = "Immediate consultation with cardiologist required. Consider preventive medications."
        
        return {
            'category': category,
            'color': color,
            'risk_percentage': risk_percentage,
            'recommendation': recommendation
        }
    
    def calculate_comprehensive_risk(self, age, sex, bp_systolic, bp_diastolic,
                                    cholesterol, hdl, smoking, diabetes,
                                    family_history=False, physical_activity=0,
                                    bmi=25, stress_level=0):
        """Comprehensive risk calculation with additional factors"""
        # Base Framingham risk
        base_score, base_percentage = self.calculate_framingham_risk(
            age, sex, bp_systolic, bp_diastolic, cholesterol, hdl, smoking, diabetes
        )
        
        # Additional factors
        additional_score = 0
        
        # Family history
        if family_history:
            additional_score += 10
        
        # Physical activity (inverse relationship)
        if physical_activity < 30:  # minutes per week
            additional_score += 10
        elif physical_activity < 150:
            additional_score += 5
        
        # BMI factor
        if bmi >= 30:
            additional_score += 10
        elif bmi >= 25:
            additional_score += 5
        
        # Stress level (0-10 scale)
        if stress_level >= 7:
            additional_score += 10
        elif stress_level >= 5:
            additional_score += 5
        
        total_score = base_score + additional_score
        total_percentage = min(total_score * 1.2, 100)
        
        return total_percentage
    
    def get_risk_factors_breakdown(self, age, sex, bp_systolic, bp_diastolic,
                                  cholesterol, hdl, smoking, diabetes,
                                  fasting_glucose=None):
        """Get detailed breakdown of risk factors"""
        factors = []
        
        # Age
        if age >= 60:
            factors.append({'factor': 'Age', 'contribution': 'High', 'value': age})
        elif age >= 50:
            factors.append({'factor': 'Age', 'contribution': 'Medium', 'value': age})
        
        # Blood Pressure
        if bp_systolic >= 140 or bp_diastolic >= 90:
            factors.append({
                'factor': 'Blood Pressure',
                'contribution': 'High',
                'value': f'{bp_systolic}/{bp_diastolic} mmHg'
            })
        elif bp_systolic >= 130 or bp_diastolic >= 85:
            factors.append({
                'factor': 'Blood Pressure',
                'contribution': 'Medium',
                'value': f'{bp_systolic}/{bp_diastolic} mmHg'
            })
        
        # Cholesterol
        if cholesterol >= 240:
            factors.append({
                'factor': 'Total Cholesterol',
                'contribution': 'High',
                'value': f'{cholesterol} mg/dL'
            })
        elif cholesterol >= 200:
            factors.append({
                'factor': 'Total Cholesterol',
                'contribution': 'Medium',
                'value': f'{cholesterol} mg/dL'
            })
        
        # HDL
        if hdl < 40:
            factors.append({
                'factor': 'HDL Cholesterol',
                'contribution': 'High Risk',
                'value': f'{hdl} mg/dL (Low)'
            })
        
        # Smoking
        if smoking == 1:
            factors.append({
                'factor': 'Smoking',
                'contribution': 'High',
                'value': 'Yes'
            })
        
        # Diabetes / fasting glucose
        if diabetes == 1:
            factors.append({
                'factor': 'Diabetes',
                'contribution': 'High',
                'value': 'Yes'
            })
        elif fasting_glucose is not None:
            if fasting_glucose >= 126:
                factors.append({
                    'factor': 'Fasting Glucose',
                    'contribution': 'High',
                    'value': f'{fasting_glucose} mg/dL (Diabetes range)'
                })
            elif fasting_glucose >= 100:
                factors.append({
                    'factor': 'Fasting Glucose',
                    'contribution': 'Medium',
                    'value': f'{fasting_glucose} mg/dL (Prediabetes range)'
                })
        
        return factors

