"""
Database Management Module
Handles user authentication and prediction history storage
"""

import sqlite3
import pandas as pd
from datetime import datetime
import json
import os


class Database:
    """Database management for users and predictions"""
    
    def __init__(self, db_path='heart_disease.db'):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction_result INTEGER NOT NULL,
                prediction_probability REAL NOT NULL,
                risk_score REAL,
                risk_category TEXT,
                features TEXT,
                model_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                age INTEGER,
                sex INTEGER,
                bmi REAL,
                bp_systolic INTEGER,
                bp_diastolic INTEGER,
                cholesterol INTEGER,
                hdl INTEGER,
                smoking INTEGER,
                diabetes INTEGER,
                physical_activity INTEGER,
                sleep_hours REAL,
                stress_level INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username, email, password_hash):
        """Create a new user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            conn.close()
            return None
    
    def get_user(self, username):
        """Get user by username"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'password_hash': user[3],
                'created_at': user[4]
            }
        return None
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'password_hash': user[3],
                'created_at': user[4]
            }
        return None
    
    def save_prediction(self, user_id, prediction_result, prediction_probability,
                       risk_score=None, risk_category=None, features=None,
                       model_used=None):
        """Save a prediction to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        features_json = json.dumps(features) if features else None
        
        cursor.execute('''
            INSERT INTO predictions 
            (user_id, prediction_result, prediction_probability, risk_score,
             risk_category, features, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, prediction_result, prediction_probability, risk_score,
              risk_category, features_json, model_used))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        return prediction_id
    
    def get_user_predictions(self, user_id, limit=50):
        """Get predictions for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        predictions = cursor.fetchall()
        conn.close()
        
        prediction_list = []
        for pred in predictions:
            features = json.loads(pred[6]) if pred[6] else None
            prediction_list.append({
                'id': pred[0],
                'user_id': pred[1],
                'prediction_result': pred[2],
                'prediction_probability': pred[3],
                'risk_score': pred[4],
                'risk_category': pred[5],
                'features': features,
                'model_used': pred[7],
                'created_at': pred[8]
            })
        
        return prediction_list
    
    def get_prediction_history_df(self, user_id):
        """Get prediction history as DataFrame"""
        predictions = self.get_user_predictions(user_id)
        
        if not predictions:
            return pd.DataFrame()
        
        df = pd.DataFrame(predictions)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        return df
    
    def save_user_profile(self, user_id, profile_data):
        """Save or update user profile"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if profile exists
        cursor.execute('SELECT id FROM user_profiles WHERE user_id = ?', (user_id,))
        exists = cursor.fetchone()
        
        if exists:
            # Update existing profile
            cursor.execute('''
                UPDATE user_profiles
                SET age = ?, sex = ?, bmi = ?, bp_systolic = ?, bp_diastolic = ?,
                    cholesterol = ?, hdl = ?, smoking = ?, diabetes = ?,
                    physical_activity = ?, sleep_hours = ?, stress_level = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (
                profile_data.get('age'),
                profile_data.get('sex'),
                profile_data.get('bmi'),
                profile_data.get('bp_systolic'),
                profile_data.get('bp_diastolic'),
                profile_data.get('cholesterol'),
                profile_data.get('hdl'),
                profile_data.get('smoking'),
                profile_data.get('diabetes'),
                profile_data.get('physical_activity'),
                profile_data.get('sleep_hours'),
                profile_data.get('stress_level'),
                user_id
            ))
        else:
            # Insert new profile
            cursor.execute('''
                INSERT INTO user_profiles
                (user_id, age, sex, bmi, bp_systolic, bp_diastolic,
                 cholesterol, hdl, smoking, diabetes, physical_activity,
                 sleep_hours, stress_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                profile_data.get('age'),
                profile_data.get('sex'),
                profile_data.get('bmi'),
                profile_data.get('bp_systolic'),
                profile_data.get('bp_diastolic'),
                profile_data.get('cholesterol'),
                profile_data.get('hdl'),
                profile_data.get('smoking'),
                profile_data.get('diabetes'),
                profile_data.get('physical_activity'),
                profile_data.get('sleep_hours'),
                profile_data.get('stress_level')
            ))
        
        conn.commit()
        conn.close()
    
    def get_user_profile(self, user_id):
        """Get user profile"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        profile = cursor.fetchone()
        conn.close()
        
        if profile:
            return {
                'id': profile[0],
                'user_id': profile[1],
                'age': profile[2],
                'sex': profile[3],
                'bmi': profile[4],
                'bp_systolic': profile[5],
                'bp_diastolic': profile[6],
                'cholesterol': profile[7],
                'hdl': profile[8],
                'smoking': profile[9],
                'diabetes': profile[10],
                'physical_activity': profile[11],
                'sleep_hours': profile[12],
                'stress_level': profile[13],
                'updated_at': profile[14]
            }
        return None
    
    def get_prediction_statistics(self, user_id):
        """Get prediction statistics for a user"""
        predictions = self.get_user_predictions(user_id, limit=1000)
        
        if not predictions:
            return None
        
        df = pd.DataFrame(predictions)
        
        stats = {
            'total_predictions': len(df),
            'average_risk_score': df['risk_score'].mean() if df['risk_score'].notna().any() else None,
            'latest_prediction': df.iloc[0] if len(df) > 0 else None,
            'risk_trend': self._calculate_risk_trend(df)
        }
        
        return stats
    
    def _calculate_risk_trend(self, df):
        """Calculate risk trend over time"""
        if len(df) < 2:
            return 'insufficient_data'
        
        df_sorted = df.sort_values('created_at')
        recent_avg = df_sorted.tail(5)['risk_score'].mean()
        older_avg = df_sorted.head(5)['risk_score'].mean()
        
        if recent_avg > older_avg + 5:
            return 'increasing'
        elif recent_avg < older_avg - 5:
            return 'decreasing'
        else:
            return 'stable'

