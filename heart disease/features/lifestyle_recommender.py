"""
Lifestyle Recommendation System
Provides personalized health recommendations based on user data
"""

import pandas as pd
import numpy as np


class LifestyleRecommender:
    """Generate personalized lifestyle recommendations"""
    
    def __init__(self):
        self.recommendations = {
            'diet': [],
            'exercise': [],
            'sleep': [],
            'stress': [],
            'monitoring': []
        }
        
    def analyze_user_profile(self, age, sex, bmi, bp_systolic, bp_diastolic,
                            cholesterol, hdl, smoking, diabetes, physical_activity,
                            sleep_hours, stress_level, family_history=False):
        """Analyze user profile and generate recommendations"""
        profile = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'bp_systolic': bp_systolic,
            'bp_diastolic': bp_diastolic,
            'cholesterol': cholesterol,
            'hdl': hdl,
            'smoking': smoking,
            'diabetes': diabetes,
            'physical_activity': physical_activity,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'family_history': family_history
        }
        
        recommendations = {
            'diet': self._get_diet_recommendations(profile),
            'exercise': self._get_exercise_recommendations(profile),
            'sleep': self._get_sleep_recommendations(profile),
            'stress': self._get_stress_recommendations(profile),
            'monitoring': self._get_monitoring_recommendations(profile),
            'general': self._get_general_recommendations(profile)
        }
        
        return recommendations
    
    def _get_diet_recommendations(self, profile):
        """Generate diet recommendations"""
        recommendations = []
        
        # Cholesterol management
        if profile['cholesterol'] >= 200:
            recommendations.append({
                'category': 'Cholesterol Control',
                'priority': 'High',
                'recommendations': [
                    'Reduce saturated fats - limit red meat, butter, and full-fat dairy',
                    'Increase soluble fiber - oats, barley, beans, apples',
                    'Include omega-3 fatty acids - fatty fish (salmon, mackerel) 2x per week',
                    'Limit trans fats - avoid processed foods and fried items',
                    'Add plant sterols - fortified foods or supplements'
                ]
            })
        
        # Blood pressure management
        if profile['bp_systolic'] >= 130 or profile['bp_diastolic'] >= 85:
            recommendations.append({
                'category': 'Blood Pressure Control',
                'priority': 'High',
                'recommendations': [
                    'Reduce sodium intake - aim for less than 2,300 mg per day',
                    'Increase potassium-rich foods - bananas, spinach, sweet potatoes',
                    'Follow DASH diet principles - fruits, vegetables, whole grains',
                    'Limit processed foods - they contain hidden sodium',
                    'Reduce alcohol consumption - limit to 1-2 drinks per day'
                ]
            })
        
        # Weight management
        if profile['bmi'] >= 25:
            recommendations.append({
                'category': 'Weight Management',
                'priority': 'High',
                'recommendations': [
                    'Create calorie deficit - reduce portion sizes by 20%',
                    'Focus on nutrient-dense foods - vegetables, lean proteins',
                    'Limit added sugars - avoid sugary drinks and desserts',
                    'Eat regular meals - don\'t skip breakfast',
                    'Stay hydrated - drink 8-10 glasses of water daily'
                ]
            })
        
        # Diabetes management
        if profile['diabetes'] == 1:
            recommendations.append({
                'category': 'Blood Sugar Control',
                'priority': 'High',
                'recommendations': [
                    'Monitor carbohydrate intake - count carbs per meal',
                    'Choose low glycemic index foods - whole grains, legumes',
                    'Eat regular, balanced meals - avoid skipping meals',
                    'Limit simple sugars - avoid candy, soda, white bread',
                    'Include protein with each meal - helps stabilize blood sugar'
                ]
            })
        
        # General heart-healthy diet
        recommendations.append({
            'category': 'Heart-Healthy Eating',
            'priority': 'Medium',
            'recommendations': [
                'Eat 5-7 servings of fruits and vegetables daily',
                'Choose whole grains over refined grains',
                'Include lean proteins - fish, poultry, legumes',
                'Use healthy fats - olive oil, avocado, nuts',
                'Limit processed meats - bacon, sausage, deli meats'
            ]
        })
        
        return recommendations
    
    def _get_exercise_recommendations(self, profile):
        """Generate exercise recommendations"""
        recommendations = []
        
        current_activity = profile['physical_activity']
        
        if current_activity < 150:  # Less than recommended 150 min/week
            recommendations.append({
                'category': 'Physical Activity Goals',
                'priority': 'High',
                'recommendations': [
                    f'Start with 30 minutes of moderate exercise, 5 days per week',
                    'Begin with walking - aim for 10,000 steps daily',
                    'Include aerobic activities - brisk walking, cycling, swimming',
                    'Add strength training - 2 days per week, 20-30 minutes',
                    'Break up sedentary time - stand every hour, take short walks'
                ],
                'weekly_goal': '150 minutes moderate or 75 minutes vigorous activity'
            })
        elif current_activity < 300:
            recommendations.append({
                'category': 'Physical Activity Enhancement',
                'priority': 'Medium',
                'recommendations': [
                    'Increase to 300 minutes of moderate activity per week',
                    'Add high-intensity interval training (HIIT) 1-2x per week',
                    'Include resistance training - build muscle mass',
                    'Try new activities - dancing, hiking, sports',
                    'Track your progress - use fitness apps or wearables'
                ]
            })
        else:
            recommendations.append({
                'category': 'Maintain Activity Level',
                'priority': 'Low',
                'recommendations': [
                    'Maintain current activity level - excellent work!',
                    'Vary your routine - prevent overuse injuries',
                    'Include recovery days - rest is important',
                    'Focus on flexibility - add yoga or stretching',
                    'Set new challenges - train for an event or goal'
                ]
            })
        
        # Age-specific recommendations
        if profile['age'] >= 65:
            recommendations.append({
                'category': 'Senior Exercise Considerations',
                'priority': 'Medium',
                'recommendations': [
                    'Focus on balance exercises - reduce fall risk',
                    'Include low-impact activities - swimming, tai chi',
                    'Consult doctor before starting new exercise program',
                    'Listen to your body - rest when needed',
                    'Stay consistent - even 10 minutes daily helps'
                ]
            })
        
        return recommendations
    
    def _get_sleep_recommendations(self, profile):
        """Generate sleep recommendations"""
        recommendations = []
        
        sleep_hours = profile['sleep_hours']
        
        if sleep_hours < 6:
            recommendations.append({
                'category': 'Sleep Improvement',
                'priority': 'High',
                'recommendations': [
                    'Aim for 7-9 hours of sleep per night',
                    'Establish consistent sleep schedule - same bedtime/wake time',
                    'Create bedtime routine - relax 1 hour before sleep',
                    'Limit screen time before bed - no devices 1 hour before sleep',
                    'Keep bedroom cool, dark, and quiet',
                    'Avoid caffeine after 2 PM',
                    'Limit alcohol - it disrupts sleep quality'
                ],
                'target_hours': 7.5
            })
        elif sleep_hours < 7:
            recommendations.append({
                'category': 'Sleep Optimization',
                'priority': 'Medium',
                'recommendations': [
                    'Increase sleep to 7-8 hours for optimal heart health',
                    'Improve sleep quality - maintain consistent schedule',
                    'Reduce evening screen time',
                    'Practice relaxation techniques before bed'
                ]
            })
        else:
            recommendations.append({
                'category': 'Maintain Healthy Sleep',
                'priority': 'Low',
                'recommendations': [
                    'Maintain current sleep schedule - excellent!',
                    'Monitor sleep quality - use sleep tracking if helpful',
                    'Keep consistent weekend schedule - avoid large variations'
                ]
            })
        
        return recommendations
    
    def _get_stress_recommendations(self, profile):
        """Generate stress management recommendations"""
        recommendations = []
        
        stress_level = profile['stress_level']  # 0-10 scale
        
        if stress_level >= 7:
            recommendations.append({
                'category': 'High Stress Management',
                'priority': 'High',
                'recommendations': [
                    'Practice daily meditation - start with 10 minutes',
                    'Try deep breathing exercises - 4-7-8 technique',
                    'Consider counseling or therapy - professional support helps',
                    'Identify stress triggers - keep a stress journal',
                    'Set boundaries - learn to say no',
                    'Take regular breaks - 5 minutes every hour',
                    'Engage in hobbies - activities you enjoy',
                    'Consider stress management programs or apps'
                ]
            })
        elif stress_level >= 5:
            recommendations.append({
                'category': 'Moderate Stress Management',
                'priority': 'Medium',
                'recommendations': [
                    'Practice mindfulness - daily meditation or yoga',
                    'Exercise regularly - physical activity reduces stress',
                    'Maintain work-life balance - set clear boundaries',
                    'Practice time management - prioritize tasks',
                    'Connect with others - social support is important',
                    'Get adequate sleep - stress and sleep are connected'
                ]
            })
        else:
            recommendations.append({
                'category': 'Maintain Low Stress',
                'priority': 'Low',
                'recommendations': [
                    'Continue current stress management practices',
                    'Maintain healthy coping strategies',
                    'Stay connected with support network'
                ]
            })
        
        return recommendations
    
    def _get_monitoring_recommendations(self, profile):
        """Generate health monitoring recommendations"""
        recommendations = []
        
        monitoring_items = []
        
        # Blood pressure monitoring
        if profile['bp_systolic'] >= 130 or profile['bp_diastolic'] >= 85:
            monitoring_items.append({
                'metric': 'Blood Pressure',
                'frequency': 'Daily',
                'target': '< 130/80 mmHg',
                'instructions': 'Measure at same time daily, rest 5 minutes before'
            })
        
        # Cholesterol monitoring
        if profile['cholesterol'] >= 200:
            monitoring_items.append({
                'metric': 'Cholesterol Panel',
                'frequency': 'Every 3-6 months',
                'target': 'Total < 200, LDL < 100, HDL > 40 (men) or > 50 (women)',
                'instructions': 'Fasting blood test required'
            })
        
        # Blood sugar monitoring
        if profile['diabetes'] == 1:
            monitoring_items.append({
                'metric': 'Blood Glucose',
                'frequency': 'As directed by doctor',
                'target': 'Fasting < 100, Post-meal < 140',
                'instructions': 'Follow diabetes management plan'
            })
        
        # Weight monitoring
        if profile['bmi'] >= 25:
            monitoring_items.append({
                'metric': 'Weight/BMI',
                'frequency': 'Weekly',
                'target': f'BMI < 25 (Current: {profile["bmi"]:.1f})',
                'instructions': 'Weigh at same time, same day, same scale'
            })
        
        # General monitoring
        monitoring_items.append({
            'metric': 'Heart Rate',
            'frequency': 'During exercise',
            'target': 'Target HR: (220 - age) × 0.6 to 0.85',
            'instructions': 'Monitor during physical activity'
        })
        
        recommendations.append({
            'category': 'Health Monitoring',
            'priority': 'High',
            'monitoring_plan': monitoring_items
        })
        
        return recommendations
    
    def _get_general_recommendations(self, profile):
        """Generate general lifestyle recommendations"""
        recommendations = []
        
        # Smoking cessation
        if profile['smoking'] == 1:
            recommendations.append({
                'category': 'Smoking Cessation',
                'priority': 'Critical',
                'recommendations': [
                    'Quit smoking immediately - #1 priority for heart health',
                    'Seek support - use quitline, apps, or support groups',
                    'Consider nicotine replacement therapy',
                    'Remove smoking triggers from environment',
                    'Celebrate milestones - each day smoke-free is progress'
                ]
            })
        
        # Alcohol moderation
        recommendations.append({
            'category': 'Alcohol Moderation',
            'priority': 'Medium',
            'recommendations': [
                'Limit alcohol to 1 drink/day (women) or 2 drinks/day (men)',
                'Avoid binge drinking - no more than 4 drinks on any occasion',
                'Choose red wine in moderation - may have heart benefits',
                'Have alcohol-free days - at least 2-3 per week'
            ]
        })
        
        # Regular check-ups
        recommendations.append({
            'category': 'Preventive Care',
            'priority': 'High',
            'recommendations': [
                'Annual physical exam with primary care doctor',
                'Cardiovascular screening every 2-5 years (or as recommended)',
                'Dental check-ups - oral health affects heart health',
                'Eye exams - diabetes and hypertension can affect vision',
                'Keep vaccinations up to date - flu, COVID-19'
            ]
        })
        
        return recommendations
    
    def generate_personalized_summary(self, profile):
        """Generate personalized summary report"""
        all_recommendations = self.analyze_user_profile(**profile)
        
        summary = {
            'user_profile': profile,
            'recommendations': all_recommendations,
            'priority_actions': self._extract_priority_actions(all_recommendations),
            'weekly_goals': self._generate_weekly_goals(profile, all_recommendations)
        }
        
        return summary
    
    def _extract_priority_actions(self, recommendations):
        """Extract high-priority actions"""
        priority_actions = []
        
        for category, items in recommendations.items():
            if isinstance(items, list):
                for item in items:
                    if item.get('priority') == 'High' or item.get('priority') == 'Critical':
                        priority_actions.extend(item.get('recommendations', [])[:3])
        
        return priority_actions[:10]  # Top 10 priority actions
    
    def _generate_weekly_goals(self, profile, recommendations):
        """Generate weekly goals"""
        goals = []
        
        # Exercise goal
        if profile['physical_activity'] < 150:
            goals.append({
                'goal': 'Exercise',
                'target': '150 minutes',
                'current': f"{profile['physical_activity']} minutes",
                'action': 'Add 30 minutes of walking, 5 days this week'
            })
        
        # Sleep goal
        if profile['sleep_hours'] < 7:
            goals.append({
                'goal': 'Sleep',
                'target': '7-8 hours/night',
                'current': f"{profile['sleep_hours']} hours",
                'action': 'Go to bed 30 minutes earlier this week'
            })
        
        # Diet goal
        goals.append({
            'goal': 'Fruits & Vegetables',
            'target': '5-7 servings daily',
            'current': 'Track this week',
            'action': 'Add 1 serving of vegetables to each meal'
        })
        
        return goals

