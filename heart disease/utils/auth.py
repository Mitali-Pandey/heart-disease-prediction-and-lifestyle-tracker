"""
Authentication Module
Handles user login and registration
"""

import bcrypt
from utils.database import Database


class Auth:
    """Authentication handler"""
    
    def __init__(self):
        self.db = Database()
    
    def hash_password(self, password):
        """Hash a password"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password, password_hash):
        """Verify a password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def register_user(self, username, email, password):
        """Register a new user"""
        # Check if user exists
        if self.db.get_user(username):
            return None, "Username already exists"
        
        # Hash password
        password_hash = self.hash_password(password)
        
        # Create user
        user_id = self.db.create_user(username, email, password_hash)
        
        if user_id:
            return user_id, "Registration successful"
        else:
            return None, "Registration failed"
    
    def login_user(self, username, password):
        """Login a user"""
        user = self.db.get_user(username)
        
        if not user:
            return None, "Username not found"
        
        if self.verify_password(password, user['password_hash']):
            return user, "Login successful"
        else:
            return None, "Incorrect password"
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        return self.db.get_user_by_id(user_id)

