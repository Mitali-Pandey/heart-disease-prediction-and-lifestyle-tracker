"""
AI Health Chatbot Module
Provides interactive explanations and health advice
"""

import re
from datetime import datetime


class HealthChatbot:
    """AI Health Chatbot for explaining predictions and providing advice"""
    
    def __init__(self):
        self.knowledge_base = {
            'cholesterol': {
                'explanation': 'Cholesterol is a waxy substance in your blood. High cholesterol can lead to plaque buildup in arteries, increasing heart disease risk.',
                'normal_range': 'Total cholesterol should be below 200 mg/dL. LDL (bad) cholesterol should be below 100 mg/dL.',
                'tips': 'Reduce saturated fats, increase fiber intake, exercise regularly, and consider medication if lifestyle changes aren\'t enough.'
            },
            'blood_pressure': {
                'explanation': 'Blood pressure measures the force of blood against artery walls. High blood pressure (hypertension) damages arteries over time.',
                'normal_range': 'Normal BP is below 120/80 mmHg. Elevated is 120-129/<80. High is 130+/80+.',
                'tips': 'Reduce sodium intake, exercise regularly, maintain healthy weight, limit alcohol, manage stress, and take prescribed medications.'
            },
            'heart_disease': {
                'explanation': 'Heart disease refers to conditions affecting the heart and blood vessels, including coronary artery disease, heart attacks, and heart failure.',
                'risk_factors': 'Age, family history, high BP, high cholesterol, diabetes, smoking, obesity, physical inactivity, and stress.',
                'prevention': 'Maintain healthy diet, exercise regularly, don\'t smoke, manage stress, control blood pressure and cholesterol, regular check-ups.'
            },
            'prediction': {
                'explanation': 'Our ML models analyze multiple factors to predict heart disease risk. Higher risk doesn\'t mean you have heart disease, but suggests increased likelihood.',
                'factors': 'Age, blood pressure, cholesterol, diabetes, smoking, family history, physical activity, and lifestyle factors.',
                'next_steps': 'Consult with a cardiologist, follow lifestyle recommendations, monitor key metrics regularly, and take preventive measures.'
            },
            'exercise': {
                'explanation': 'Regular exercise strengthens the heart, improves circulation, lowers blood pressure, and helps maintain healthy weight.',
                'recommendation': 'Aim for 150 minutes of moderate-intensity exercise or 75 minutes of vigorous exercise per week.',
                'types': 'Aerobic exercises (walking, running, swimming), strength training, and flexibility exercises.'
            },
            'diet': {
                'explanation': 'A heart-healthy diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats while limiting saturated fats, trans fats, sodium, and added sugars.',
                'recommendation': 'Follow DASH or Mediterranean diet patterns. Focus on plant-based foods, limit processed foods.',
                'foods': 'Salmon, oats, berries, leafy greens, nuts, beans, whole grains, olive oil, and dark chocolate (in moderation).'
            }
        }
    
    def process_query(self, query, context=None):
        """Process user query and generate response"""
        query_lower = query.lower()
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Generate response based on intent
        response = self._generate_response(intent, query_lower, context)
        
        return response
    
    def _detect_intent(self, query):
        """Detect user intent from query"""
        # Greeting
        if any(word in query for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting'
        
        # Explanation requests
        if any(word in query for word in ['explain', 'what is', 'tell me about', 'how does', 'why']):
            return 'explanation'
        
        # Prediction questions
        if any(word in query for word in ['prediction', 'result', 'risk', 'probability', 'chance']):
            return 'prediction'
        
        # Advice requests
        if any(word in query for word in ['advice', 'recommend', 'suggest', 'should i', 'what should']):
            return 'advice'
        
        # Specific topic queries
        topics = ['cholesterol', 'blood pressure', 'bp', 'exercise', 'diet', 'heart disease', 'smoking', 'diabetes']
        for topic in topics:
            if topic in query:
                return f'topic_{topic.replace(" ", "_")}'
        
        # Default
        return 'general'
    
    def _generate_response(self, intent, query, context=None):
        """Generate response based on intent"""
        if intent == 'greeting':
            return {
                'response': "Hello! I'm your AI health assistant. I can help explain your heart disease prediction results, answer questions about risk factors, and provide health recommendations. How can I help you today?",
                'suggestions': [
                    "Explain my prediction results",
                    "What is cholesterol?",
                    "How can I reduce my risk?",
                    "Tell me about blood pressure"
                ]
            }
        
        elif intent == 'explanation':
            # Extract topic
            topic = self._extract_topic(query)
            if topic and topic in self.knowledge_base:
                kb = self.knowledge_base[topic]
                return {
                    'response': f"{kb['explanation']}\n\nNormal Range: {kb.get('normal_range', 'N/A')}\n\nTips: {kb.get('tips', 'N/A')}",
                    'topic': topic
                }
            else:
                return {
                    'response': "I can explain various topics related to heart health including cholesterol, blood pressure, heart disease, exercise, diet, and predictions. What would you like to know more about?",
                    'suggestions': ['cholesterol', 'blood pressure', 'heart disease', 'exercise', 'diet']
                }
        
        elif intent == 'prediction':
            if context and 'prediction_result' in context:
                pred = context['prediction_result']
                prob = pred.get('probability', 0) * 100
                risk_cat = context.get('risk_category', 'Unknown')
                
                response_text = f"Your prediction shows a {prob:.1f}% probability of heart disease risk, which is categorized as {risk_cat}.\n\n"
                response_text += "This prediction is based on multiple factors including age, blood pressure, cholesterol, and lifestyle factors. "
                response_text += "Higher risk doesn't mean you currently have heart disease, but suggests you should take preventive measures.\n\n"
                response_text += "I recommend consulting with a cardiologist and following the lifestyle recommendations provided."
                
                return {
                    'response': response_text,
                    'context_used': True
                }
            else:
                return {
                    'response': "I'd be happy to explain your prediction results! However, I need your prediction data to provide specific information. Please make a prediction first, or ask me general questions about heart disease risk factors.",
                    'suggestions': ['What factors affect heart disease risk?', 'How is risk calculated?']
                }
        
        elif intent == 'advice':
            if context and 'risk_category' in context:
                risk_cat = context['risk_category']
                
                if 'high' in risk_cat.lower():
                    advice = "Given your high risk level, I strongly recommend:\n\n"
                    advice += "1. Consult with a cardiologist immediately\n"
                    advice += "2. Follow all medical recommendations\n"
                    advice += "3. Make immediate lifestyle changes\n"
                    advice += "4. Monitor your health metrics regularly\n"
                    advice += "5. Consider preventive medications if recommended"
                elif 'medium' in risk_cat.lower():
                    advice = "Given your medium risk level, I recommend:\n\n"
                    advice += "1. Schedule an appointment with your doctor\n"
                    advice += "2. Implement lifestyle modifications\n"
                    advice += "3. Monitor blood pressure and cholesterol\n"
                    advice += "4. Increase physical activity\n"
                    advice += "5. Follow a heart-healthy diet"
                else:
                    advice = "Given your low risk level, maintain your healthy habits:\n\n"
                    advice += "1. Continue regular exercise\n"
                    advice += "2. Maintain healthy diet\n"
                    advice += "3. Get regular check-ups\n"
                    advice += "4. Monitor key health metrics\n"
                    advice += "5. Stay informed about heart health"
                
                return {
                    'response': advice,
                    'context_used': True
                }
            else:
                return {
                    'response': "I can provide personalized advice based on your risk assessment. General recommendations include:\n\n"
                              "• Maintain healthy weight\n"
                              "• Exercise regularly (150 min/week)\n"
                              "• Eat heart-healthy diet\n"
                              "• Don't smoke\n"
                              "• Manage stress\n"
                              "• Control blood pressure and cholesterol\n"
                              "• Get regular check-ups",
                    'suggestions': ['How much exercise do I need?', 'What is a heart-healthy diet?']
                }
        
        elif intent.startswith('topic_'):
            topic = intent.replace('topic_', '').replace('_', ' ')
            if topic in self.knowledge_base:
                kb = self.knowledge_base[topic]
                return {
                    'response': f"{kb['explanation']}\n\n{kb.get('tips', '')}",
                    'topic': topic
                }
        
        # Default response
        return {
            'response': "I'm here to help with questions about heart health, your prediction results, and lifestyle recommendations. You can ask me about:\n\n"
                       "• Your prediction results\n"
                       "• Risk factors (cholesterol, blood pressure, etc.)\n"
                       "• Lifestyle recommendations\n"
                       "• General heart health information\n\n"
                       "What would you like to know?",
            'suggestions': [
                "Explain my prediction",
                "What is high cholesterol?",
                "How can I reduce my risk?",
                "Tell me about exercise for heart health"
            ]
        }
    
    def _extract_topic(self, query):
        """Extract topic from query"""
        topics_map = {
            'cholesterol': 'cholesterol',
            'blood pressure': 'blood_pressure',
            'bp': 'blood_pressure',
            'heart disease': 'heart_disease',
            'exercise': 'exercise',
            'diet': 'diet',
            'prediction': 'prediction',
            'risk': 'prediction'
        }
        
        for key, value in topics_map.items():
            if key in query:
                return value
        
        return None
    
    def explain_prediction(self, prediction_result, risk_assessment):
        """Provide detailed explanation of prediction"""
        prob = prediction_result.get('probability', 0) * 100
        risk_cat = risk_assessment.get('category', 'Unknown')
        
        explanation = f"""
**Prediction Explanation:**

Your heart disease risk assessment shows a **{prob:.1f}% probability**, which is categorized as **{risk_cat}**.

**What this means:**
- This prediction is based on analysis of multiple risk factors using advanced machine learning models
- Higher probability indicates increased likelihood, not a diagnosis
- Many risk factors can be modified through lifestyle changes

**Key Factors Considered:**
- Age and biological factors
- Blood pressure levels
- Cholesterol levels
- Lifestyle factors (smoking, exercise, diet)
- Medical history (diabetes, family history)

**Next Steps:**
{risk_assessment.get('recommendation', 'Consult with healthcare provider')}

**Remember:** This tool is for educational purposes. Always consult with qualified healthcare professionals for medical decisions.
        """
        
        return explanation
    
    def get_contextual_suggestions(self, context=None):
        """Get contextual suggestions based on current state"""
        if context and 'risk_category' in context:
            risk_cat = context['risk_category'].lower()
            
            if 'high' in risk_cat:
                return [
                    "What should I do about my high risk?",
                    "Explain my prediction results",
                    "What lifestyle changes are most important?",
                    "How often should I get checked?"
                ]
            elif 'medium' in risk_cat:
                return [
                    "How can I reduce my risk?",
                    "What is a heart-healthy diet?",
                    "How much exercise do I need?",
                    "Explain cholesterol"
                ]
            else:
                return [
                    "How do I maintain low risk?",
                    "What is preventive care?",
                    "Tell me about heart-healthy habits",
                    "What should I monitor?"
                ]
        
        return [
            "What is heart disease?",
            "How is risk calculated?",
            "What are the main risk factors?",
            "How can I improve my heart health?"
        ]

