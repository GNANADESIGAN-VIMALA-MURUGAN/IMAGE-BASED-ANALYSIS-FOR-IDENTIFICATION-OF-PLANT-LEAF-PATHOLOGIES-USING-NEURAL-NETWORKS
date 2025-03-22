from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import sqlite3
import base64
from werkzeug.utils import secure_filename
from functools import wraps
import uuid  # For generating unique group IDs
import webbrowser
import threading
from datetime import datetime
import pytz
import re

# Import functions from the respective model files
from cnn_test import test1 as cnn_test
from rnn_test import test_model as rnn_test
from ann_test import test_model as ann_test

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_predicted_class(predicted_class):
    """
    Normalize the predicted class to match the keys in the disease_descriptions dictionary.
    Example: "Potato___Early_Blight" -> "Early Blight"
    """
    parts = predicted_class.split("___")
    if len(parts) > 1:
        return parts[1].replace("_", " ")
    return predicted_class.replace("_", " ")


disease_descriptions = {
    "tomato": {
        "healthy": "The tomato leaf is healthy and shows no signs of disease.",
        "early blight": " Use fungicides like chlorothalonil, practice crop rotation, and remove infected leaves.",
        "late blight":" Apply fungicides containing copper or mancozeb, ensure proper drainage, and destroy infected plants."
    },
    "potato": {
        "healthy": "The potato leaf is healthy and shows no signs of disease.",
        "early blight":"Use resistant varieties, apply fungicides, and ensure proper spacing for airflow.",
        "late blight": "Spray fungicides early, avoid overhead irrigation, and remove volunteer plants."
    },
    "grape": {
        "healthy": "The grape leaf is healthy and shows no signs of disease.",
        "black rot": "Prune infected areas, use fungicides like myclobutanil, and improve air circulation.",
        "black measles": "Apply copper-based fungicides, prune affected parts, and maintain vineyard hygiene."
    }
}


# Configuration
USERS_DB = 'users.db'  # Database for user authentication
PREDICTIONS_DB = 'plant_disease.db'  # Database for predictions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ADMIN_DB = 'admin.db'

# Initialize SQLite database for users
def init_users_db():
    conn = sqlite3.connect(USERS_DB)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

# Initialize SQLite database for Admin
def init_admin_db():
    conn = sqlite3.connect(ADMIN_DB)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS admin_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

# Initialize SQLite database for predictions
def init_predictions_db():
    conn = sqlite3.connect(PREDICTIONS_DB)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        leaf_type TEXT NOT NULL,
        model_type TEXT NOT NULL,
        predicted_class TEXT,
        confidence REAL,
        prediction_time REAL,
        image BLOB NOT NULL,
        group_id TEXT DEFAULT NULL,  -- New column to group predictions
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Add a timestamp column
    )
    ''')
    conn.commit()
    conn.close()

# Call initialization functions
init_users_db()
init_predictions_db()
init_admin_db()

# Decorator to protect routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("You must be logged in to access this page.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("You must be logged in as an admin to access this page.", "error")
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Validation helper functions
def is_username_valid(username):
    # 3-20 characters, letters/numbers/underscores only
    pattern = r"^[a-zA-Z0-9_]{3,20}$"
    return re.match(pattern, username) is not None

def is_email_valid(email):
    # Basic email format validation
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None


def is_password_strong(password):
    # Minimum 8 characters, at least two types of characters (letters, numbers, symbols)
    if len(password) < 8:
        return False
    
    has_letter = bool(re.search(r'[A-Za-z]', password))  # Check for letters
    has_number = bool(re.search(r'\d', password))       # Check for numbers
    has_symbol = bool(re.search(r'[@$!%*?&]', password))  # Check for symbols
    
    # Count how many types of characters are present
    type_count = sum([has_letter, has_number, has_symbol])
    
    # Require at least two types of characters
    return type_count >= 2


# Regular user signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check all fields are filled
        if not all([username, email, password]):
            flash("All fields are required.", "error")
            return redirect(url_for('signup'))
        
        # Validate username
        if not is_username_valid(username):
            flash("Username must be 3-20 characters with only letters, numbers, or underscores", "error")
            return redirect(url_for('signup'))
        
        # Validate email
        if not is_email_valid(email):
            flash("Please enter a valid email address", "error")
            return redirect(url_for('signup'))
        
        # Check password strength
        if not is_password_strong(password):
            flash("Password must be at least 8 characters with: 1 uppercase, 1 lowercase, 1 number, and 1 special character (@$!%*?&)", "error")
            return redirect(url_for('signup'))
        
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', 
                          (username, email, password))
            conn.commit()
            conn.close()
            flash("Signup successful! Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")
            return redirect(url_for('signup'))

    
    return render_template('signup.html')

# Admin signup
@app.route('/admin/signup', methods=['GET', 'POST'])
def admin_signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not all([username, email, password]):
            flash("All fields are required.", "error")
            return redirect(url_for('admin_signup'))
        
        if not is_username_valid(username):
            flash("Username must be 3-20 characters with only letters, numbers, or underscores", "error")
            return redirect(url_for('admin_signup'))
        
        if not is_email_valid(email):
            flash("Please enter a valid email address", "error")
            return redirect(url_for('admin_signup'))
        
        if not is_password_strong(password):
            flash("Password must be at least 8 characters with: 1 uppercase, 1 lowercase, 1 number, and 1 special character (@$!%*?&)", "error")
            return redirect(url_for('admin_signup'))
        
        try:
            conn = sqlite3.connect('admin.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO admin_users (username, email, password) VALUES (?, ?, ?)', 
                          (username, email, password))
            conn.commit()
            conn.close()
            flash("Signup successful! Please log in.", "success")
            return redirect(url_for('admin_signup'))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")
            return redirect(url_for('admin_signup'))

    
    return render_template('admin_signup.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form.get('identifier')  # Can be username or email
        password = request.form.get('password')
        if not identifier or not password:
            flash("All fields are required.", "error")
            return redirect(url_for('login'))
        try:
            conn = sqlite3.connect(USERS_DB)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE (username = ? OR email = ?) AND password = ?', (identifier, identifier, password))
            user = cursor.fetchone()
            conn.close()
            if user:
                session['user_id'] = user[0]
                
                return redirect(url_for('index'))
            else:
                flash("Invalid username/email or password.", "error")
                return redirect(url_for('login'))
        except Exception as e:
            flash(f"An unexpected error occurred: {str(e)}", "error")
            return redirect(url_for('login'))
    return render_template('login.html')

# Admin Login Page
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        identifier = request.form.get('identifier')  # Can be username or email
        password = request.form.get('password')
        if not identifier or not password:
            flash("All fields are required.", "error")
            return redirect(url_for('admin_login'))
        try:
            conn = sqlite3.connect(ADMIN_DB)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM admin_users WHERE (username = ? OR email = ?) AND password = ?', (identifier, identifier, password))
            user = cursor.fetchone()
            conn.close()
            if user:
                session['user_id'] = user[0]
                return redirect(url_for('admin_dashboard'))
            else:
                flash("Invalid username/email or password.", "error")
                return redirect(url_for('admin_login'))
        except Exception as e:
            flash(f"An unexpected error occurred: {str(e)}", "error")
            return redirect(url_for('admin_login'))
    return render_template('admin_login.html')

# Admin Dashboard
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html')

# Admin User Data Page
@app.route('/admin/users')
@admin_required
def admin_users():
    try:
        conn = sqlite3.connect(USERS_DB)
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email, password FROM users')
        users = cursor.fetchall()
        conn.close()
        return render_template('admin_users.html', users=users)
    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}", "error")
        return redirect(url_for('admin_dashboard'))

# Admin History Page
@app.route('/admin/history')
@admin_required
def admin_history():
    try:
        conn = sqlite3.connect(PREDICTIONS_DB)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT group_id, leaf_type, model_type, predicted_class, confidence, prediction_time, created_at 
        FROM predictions
        ORDER BY created_at ASC
        ''')
        rows = cursor.fetchall()
        conn.close()

        group_counts = {}
        for row in rows:
            group_id = row[0]
            if group_id:
                group_counts[group_id] = group_counts.get(group_id, 0) + 1

        predictions = []
        seen_group_ids = set()
        for row in rows:
            group_id, leaf_type, model_type, predicted_class, confidence, prediction_time, created_at = row

            utc_time = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            local_timezone = pytz.timezone('Asia/Kolkata')  # Replace with your actual timezone
            local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(local_timezone)
            formatted_time = local_time.strftime('%b %d, %Y %I:%M:%S %p')

            if group_id and group_counts.get(group_id, 1) > 1:
                if group_id not in seen_group_ids:
                    predictions.append({
                        'id': group_id,
                        'leaf_type': leaf_type,
                        'model_type': 'All Models',
                        'predicted_class': 'Multiple Classes',
                        'confidence': 'Varies',
                        'prediction_time': 'Varies',
                        'created_at': formatted_time
                    })
                    seen_group_ids.add(group_id)
            else:
                predictions.append({
                    'id': group_id or str(uuid.uuid4()),
                    'leaf_type': leaf_type,
                    'model_type': model_type,
                    'predicted_class': predicted_class,
                    'confidence': f"{confidence:.2f}%",
                    'prediction_time': prediction_time,
                    'created_at': formatted_time
                })

        return render_template('admin_history.html', predictions=predictions)

    except Exception as e:
        return render_template('admin_history.html', error=f"An unexpected error occurred: {str(e)}")

# Index Page (Protected Route)
@app.route('/index')
@login_required
def index():
    return render_template('index.html')

# Logout
@app.route('/logout')
def logout():
    session.clear()  # Clears all session data, including flash messages
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))



@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return render_template('result.html', error="No file uploaded.")
    file = request.files['file']
    leaf_type = request.form.get('leaf_type')  # Plant type (e.g., potato, tomato, grape)
    model_type = request.form.get('model_type')

    if leaf_type not in ['potato', 'tomato', 'grape']:
        return render_template('result.html', error="Invalid leaf type selected.")
    if model_type not in ['cnn', 'RNN_GRU', 'ANN_MLP', 'all']:
        return render_template('result.html', error="Invalid model type selected.")
    if file.filename == '':
        return render_template('result.html', error="No selected file.")
    if not allowed_file(file.filename):
        return render_template('result.html', error="Invalid file type. Only PNG, JPG, and JPEG files are allowed.")
    if len(file.read()) > MAX_FILE_SIZE:
        return render_template('result.html', error="File size exceeds 5MB. Please upload a smaller image.")
    file.seek(0)

    try:
        file_data = file.read()
        group_id = str(uuid.uuid4())

        if model_type == 'all':
            cnn_result = cnn_test(leaf_type, file_data)
            RNN_GRU_result = rnn_test(leaf_type, file_data)
            ANN_MLP_result = ann_test(leaf_type, file_data)

            if 'error' in cnn_result or 'error' in RNN_GRU_result or 'error' in ANN_MLP_result:
                error_message = (
                    cnn_result.get('error', '') +
                    RNN_GRU_result.get('error', '') +
                    ANN_MLP_result.get('error', '')
                )
                return render_template('result.html', error=error_message)


            # Normalize predicted classes
            cnn_result['predicted_class'] = normalize_predicted_class(cnn_result['predicted_class'])
            RNN_GRU_result['predicted_class'] = normalize_predicted_class(RNN_GRU_result['predicted_class'])
            ANN_MLP_result['predicted_class'] = normalize_predicted_class(ANN_MLP_result['predicted_class'])

            conn = sqlite3.connect(PREDICTIONS_DB)
            cursor = conn.cursor()
            for model, result in [('cnn', cnn_result), ('RNN_GRU', RNN_GRU_result), ('ANN_MLP', ANN_MLP_result)]:
                cursor.execute('''
                INSERT INTO predictions (leaf_type, model_type, predicted_class, confidence, prediction_time, image, group_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (leaf_type, model, result['predicted_class'], result['confidence'], result['prediction_time'], file_data, group_id))
            conn.commit()
            conn.close()

            return redirect(url_for('show_result', image_id=group_id))

        else:
            if model_type == 'cnn':
                result = cnn_test(leaf_type, file_data)
            elif model_type == 'RNN_GRU':
                result = rnn_test(leaf_type, file_data)
            elif model_type == 'ANN_MLP':
                result = ann_test(leaf_type, file_data)

            if 'error' in result:
                return render_template('result.html', error=result['error'])

            # Normalize predicted class
            result['predicted_class'] = normalize_predicted_class(result['predicted_class'])

            conn = sqlite3.connect(PREDICTIONS_DB)
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO predictions (leaf_type, model_type, predicted_class, confidence, prediction_time, image, group_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (leaf_type, model_type, result['predicted_class'], result['confidence'], result['prediction_time'], file_data, group_id))
            conn.commit()
            conn.close()

            return redirect(url_for('show_result', image_id=group_id))

    except Exception as e:
        return render_template('result.html', error=f"An unexpected error occurred: {str(e)}")
    

# Result Page
@app.route('/result/<image_id>')
@login_required
def show_result(image_id):
    try:
        conn = sqlite3.connect(PREDICTIONS_DB)
        cursor = conn.cursor()

        # Fetch the image using group_id
        cursor.execute('SELECT image FROM predictions WHERE group_id = ?', (image_id,))
        row = cursor.fetchone()
        if not row:
            return render_template('result.html', error="Image not found.")

        image_data = row[0]
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Fetch all predictions for the same group_id
        cursor.execute('''
        SELECT leaf_type,model_type, predicted_class, confidence, prediction_time 
        FROM predictions 
        WHERE group_id = ?
        ''', (image_id,))
        rows = cursor.fetchall()
        conn.close()

        results = {}
        for row in rows:
            leaf_type,model_type, predicted_class, confidence, prediction_time = row
            results[model_type] = {
                'leaf_type':leaf_type,
                'predicted_class': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'prediction_time': f"{prediction_time:.2f} seconds",
            }
            
        if len(results) == 3:  # CNN, RNN_GRU, ANN_MLP
            return render_template(
                'result.html',
                image_data=image_base64,
                cnn_result=results.get('cnn'),
                RNN_GRU_result=results.get('RNN_GRU'),
                ANN_MLP_result=results.get('ANN_MLP'),
                disease_descriptions=disease_descriptions  # Pass the descriptions
            )
        
        


        model_type = list(results.keys())[0]
        print(model_type)
        result = results[model_type]
        return render_template(
            'result.html',
            image_data=image_base64,
            plant_name=leaf_type,
            predicted_label=result['predicted_class'],
            confidence=result['confidence'],
            prediction_time=result['prediction_time'],
            model_type=model_type.upper(),
            disease_descriptions=disease_descriptions  # Pass the descriptions
        )

    except Exception as e:
        return render_template('result.html', error=f"An unexpected error occurred: {str(e)}")

#admin result
@app.route('/admin/result/<image_id>')
@admin_required
def ashow_result(image_id):
    try:
        conn = sqlite3.connect(PREDICTIONS_DB)
        cursor = conn.cursor()

        # Fetch the image using group_id
        cursor.execute('SELECT image FROM predictions WHERE group_id = ?', (image_id,))
        row = cursor.fetchone()
        if not row:
            return render_template('admin_result.html', error="Image not found.")

        image_data = row[0]
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Fetch all predictions for the same group_id
        cursor.execute('''
        SELECT leaf_type,model_type, predicted_class, confidence, prediction_time 
        FROM predictions 
        WHERE group_id = ?
        ''', (image_id,))
        rows = cursor.fetchall()
        conn.close()

        results = {}
        for row in rows:
            leaf_type,model_type, predicted_class, confidence, prediction_time = row
            results[model_type] = {
                'leaf_type':leaf_type,
                'predicted_class': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'prediction_time': f"{prediction_time:.2f} seconds",
            }
            
        if len(results) == 3:  # CNN, RNN_GRU, ANN_MLP
            return render_template(
                'admin_result.html',
                image_data=image_base64,
                cnn_result=results.get('cnn'),
                RNN_GRU_result=results.get('RNN_GRU'),
                ANN_MLP_result=results.get('ANN_MLP'),
                disease_descriptions=disease_descriptions  # Pass the descriptions
            )
        
        


        model_type = list(results.keys())[0]
        print(model_type)
        result = results[model_type]
        return render_template(
            'admin_result.html',
            image_data=image_base64,
            plant_name=leaf_type,
            predicted_label=result['predicted_class'],
            confidence=result['confidence'],
            prediction_time=result['prediction_time'],
            model_type=model_type.upper(),
            disease_descriptions=disease_descriptions  # Pass the descriptions
        )

    except Exception as e:
        return render_template('admin_result.html', error=f"An unexpected error occurred: {str(e)}")

# History Page
@app.route('/history')
@login_required
def history():
    try:
        conn = sqlite3.connect(PREDICTIONS_DB)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT group_id, leaf_type, model_type, predicted_class, confidence, prediction_time, created_at 
        FROM predictions
        ORDER BY created_at ASC
        ''')
        rows = cursor.fetchall()
        conn.close()

        group_counts = {}
        for row in rows:
            group_id = row[0]
            if group_id:
                group_counts[group_id] = group_counts.get(group_id, 0) + 1

        predictions = []
        seen_group_ids = set()
        for row in rows:
            group_id, leaf_type, model_type, predicted_class, confidence, prediction_time, created_at = row

            utc_time = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            local_timezone = pytz.timezone('Asia/Kolkata')  # Replace with your actual timezone
            local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(local_timezone)
            formatted_time = local_time.strftime('%b %d, %Y %I:%M:%S %p')

            if group_id and group_counts.get(group_id, 1) > 1:
                if group_id not in seen_group_ids:
                    predictions.append({
                        'id': group_id,
                        'leaf_type': leaf_type,
                        'model_type': 'All Models',
                        'predicted_class': 'Multiple Classes',
                        'confidence': 'Varies',
                        'prediction_time': 'Varies',
                        'created_at': formatted_time
                    })
                    seen_group_ids.add(group_id)
            else:
                predictions.append({
                    'id': group_id or str(uuid.uuid4()),
                    'leaf_type': leaf_type,
                    'model_type': model_type,
                    'predicted_class': predicted_class,
                    'confidence': f"{confidence:.2f}%",
                    'prediction_time': prediction_time,
                    'created_at': formatted_time
                })

        return render_template('history.html', predictions=predictions)

    except Exception as e:
        return render_template('history.html', error=f"An unexpected error occurred: {str(e)}")

# View Specific Prediction
@app.route('/history/<prediction_id>')
@login_required
def view_prediction(prediction_id):
    try:
        conn = sqlite3.connect(PREDICTIONS_DB)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT image,leaf_type, model_type, predicted_class, confidence, prediction_time 
        FROM predictions 
        WHERE group_id = ?
        ''', (prediction_id,))
        rows = cursor.fetchall()

        if not rows:
            return render_template('result.html', error="Prediction not found.")

        image_data = rows[0][0]
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        results = {}
        for row in rows:
            _,leaf_type, model_type, predicted_class, confidence, prediction_time = row
            results[model_type] = {
                'predicted_class': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'prediction_time': f"{prediction_time:.2f} seconds",
                'leaf_type': leaf_type,
            }

        if len(results) == 3:  # CNN, RNN_GRU, ANN_MLP
            return render_template(
                'result.html',
                image_data=image_base64,
                cnn_result=results.get('cnn'),
                RNN_GRU_result=results.get('RNN_GRU'),
                ANN_MLP_result=results.get('ANN_MLP'),
                disease_descriptions=disease_descriptions  # Pass the descriptions
            )

        model_type = list(results.keys())[0]
        result = results[model_type]
        return render_template(
            'result.html',
            image_data=image_base64,
            plant_name=leaf_type,
            predicted_label=result['predicted_class'],
            confidence=result['confidence'],
            prediction_time=result['prediction_time'],
            model_type=model_type.upper(),
            disease_descriptions=disease_descriptions  # Pass the descriptions
        )

    except Exception as e:
        return render_template('result.html', error=f"An unexpected error occurred: {str(e)}")

# Delete Prediction
@app.route('/delete/<prediction_id>', methods=['POST'])
@admin_required
def delete_prediction(prediction_id):
    try:
        conn = sqlite3.connect(PREDICTIONS_DB)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions WHERE group_id = ?', (prediction_id,))
        conn.commit()
        conn.close()
        flash("Prediction deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting prediction: {str(e)}", "error")
    return redirect(url_for('admin_history'))

# Clear All Predictions
@app.route('/clear', methods=['POST'])
@admin_required
def clear_predictions():
    try:
        conn = sqlite3.connect(PREDICTIONS_DB)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions')
        conn.commit()
        conn.close()
        flash("All predictions cleared successfully.", "success")
    except Exception as e:
        flash(f"Error clearing predictions: {str(e)}", "error")
    return redirect(url_for('admin_history'))

# Delete User
@app.route('/user_delete/<user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    try:
        conn = sqlite3.connect(USERS_DB)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        flash("User deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting user: {str(e)}", "error")
    return redirect(url_for('admin_users'))

# Clear All Users
@app.route('/user_clear', methods=['POST'])
@admin_required
def admin_clear_users():
    try:
        conn = sqlite3.connect(USERS_DB)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users')
        conn.commit()
        conn.close()
        flash("All users cleared successfully.", "success")
    except Exception as e:
        flash(f"Error clearing users: {str(e)}", "error")
    return redirect(url_for('admin_users'))
####
# Admin User Data Page
@app.route('/admin/users')
@admin_required
def aadmin_users():
    try:
        conn = sqlite3.connect(USERS_DB)
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email FROM users')
        users = cursor.fetchall()
        conn.close()
        return render_template('admin_users.html', users=users)
    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}", "error")
        return redirect(url_for('admin_dashboard'))

# Admin Admin Data Page
@app.route('/admin/admins')
@admin_required
def admin_admins():
    try:
        conn = sqlite3.connect(ADMIN_DB)
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email FROM admin_users')
        admins = cursor.fetchall()
        conn.close()
        return render_template('admin_admins.html', admins=admins)
    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}", "error")
        return redirect(url_for('admin_dashboard'))

# Delete Admin
@app.route('/admin/delete/<int:admin_id>', methods=['POST'])
@admin_required
def admin_delete_admin(admin_id):
    try:
        conn = sqlite3.connect(ADMIN_DB)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM admin_users WHERE id = ?', (admin_id,))
        conn.commit()
        conn.close()
        flash("Admin deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting admin: {str(e)}", "error")
    return redirect(url_for('admin_admins'))
# Clear All Admins
@app.route('/admin/clear', methods=['POST'])
@admin_required
def admin_clear_admins():
    try:
        conn = sqlite3.connect(ADMIN_DB)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM admin_users')
        conn.commit()
        conn.close()
        flash("All admin accounts cleared successfully.", "success")
    except Exception as e:
        flash(f"Error clearing admin accounts: {str(e)}", "error")
    return redirect(url_for('admin_admins'))
# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    if not hasattr(open_browser, "has_run"):  # Prevent multiple openings
        open_browser.has_run = True
        threading.Timer(1.5, open_browser).start()  # Delay for smooth opening
    app.run(debug=True, use_reloader=False)

   # (history not working)