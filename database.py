import sqlite3
import bcrypt


class UserDatabase:
    def __init__(self, db_path='users.db'):
        """Initialize database connection and create users table if not exists"""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # Create users table if not exists
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                email TEXT UNIQUE
            )
        ''')
        self.conn.commit()

    def hash_password(self, password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    def verify_password(self, stored_password, provided_password):
        """Verify provided password against stored hash"""
        return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

    def register_user(self, username, password, email):
        """Register a new user"""
        try:
            hashed_password = self.hash_password(password)
            self.cursor.execute(
                'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                (username, hashed_password, email)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def login_user(self, username, password):
        """Authenticate user"""
        self.cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
        result = self.cursor.fetchone()

        if result:
            stored_password = result[0]
            return self.verify_password(stored_password, password)
        return False

    def user_exists(self, username):
        """Check if user exists"""
        self.cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        return self.cursor.fetchone() is not None

    def close(self):
        """Close database connection"""
        self.conn.close()