import os

# Define the file contents
APP_PY = r"""import os
import cv2
import numpy as np
import time
from PIL import Image
from google import genai
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scavenger.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None

class Hunt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    target_image = db.Column(db.String(255), nullable=False)
    clue1 = db.Column(db.String(255), nullable=False)
    clue2 = db.Column(db.String(255), nullable=False)
    clue3 = db.Column(db.String(255), nullable=False)
    attempts = db.relationship('Attempt', backref='hunt', lazy=True)

class Attempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hunt_id = db.Column(db.Integer, db.ForeignKey('hunt.id'), nullable=False)
    attempt_image = db.Column(db.String(255), nullable=False)
    clues_used = db.Column(db.Integer, nullable=False)
    score = db.Column(db.Integer, nullable=False)
    is_successful = db.Column(db.Boolean, nullable=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_score(clues_used, is_successful):
    if not is_successful: return 0
    if clues_used == 1: return 100
    elif clues_used == 2: return 50
    elif clues_used >= 3: return 25
    return 0

def verify_match(target_path, attempt_path, min_match_count=12, min_inliers=8, ratio_thresh=0.75):
    # Use Gemini API for smart, semantic image matching if an API key is provided
    if client:
        try:
            target_img = Image.open(target_path)
            attempt_img = Image.open(attempt_path)
            
            prompt = (
                "You are an AI judge for a photo scavenger hunt. "
                "I am providing you with two images: a reference target image and a player's attempt. "
                "The player's attempt might have different lighting, different angles, zoom levels, "
                "or contain people/objects not in the original. "
                "Determine if the player successfully photographed the same specific item or location. "
                "Answer ONLY with 'True' if it is a valid match, or 'False' if it is not."
            )
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[prompt, target_img, attempt_img]
                    )
                    text_response = response.text.strip().lower()
                    
                    print(f"DEBUG: Gemini Vision API result: {response.text}")
                    if 'true' in text_response:
                        return True
                    return False
                except Exception as e:
                    error_str = str(e)
                    if ('503' in error_str or '429' in error_str or 'UNAVAILABLE' in error_str) and attempt < max_retries - 1:
                        sleep_time = 2 ** attempt
                        print(f"Gemini API busy (503/429). Retrying in {sleep_time} seconds (Attempt {attempt + 1}/{max_retries})...")
                        time.sleep(sleep_time)
                    else:
                        print(f"Error using Gemini API: {e}")
                        print("Falling back to OpenCV...")
                        break
        except Exception as e:
            print(f"Error loading images for Gemini API: {e}")
            print("Falling back to OpenCV...")

    try:
        img1 = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(attempt_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None: return False

        def resize_img(img, max_dim=800):
            h, w = img.shape
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                return cv2.resize(img, (int(w * scale), int(h * scale)))
            return img

        img1 = resize_img(img1)
        img2 = resize_img(img2)

        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2: return False

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        
        print(f"DEBUG: Found {len(good_matches)} good matches before geometric check.")
        
        if len(good_matches) >= min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is not None:
                inliers = int(np.sum(mask))
                print(f"DEBUG: RANSAC geometric check found {inliers} inliers.")
                return inliers >= min_inliers
            else:
                print("DEBUG: Homography could not be calculated (no geometric match).")
                return False
        else:
            print(f"DEBUG: Not enough matches found ({len(good_matches)}/{min_match_count}).")
            return False
    except Exception as e:
        print(f"Error during image matching: {e}")
        return False

@app.route('/')
def index():
    hunts = Hunt.query.all()
    return render_template('index.html', hunts=hunts)

@app.route('/create', methods=['GET', 'POST'])
def create_hunt():
    if request.method == 'POST':
        if 'target_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['target_image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            clue1 = request.form.get('clue1')
            clue2 = request.form.get('clue2')
            clue3 = request.form.get('clue3')
            
            new_hunt = Hunt(target_image=filename, clue1=clue1, clue2=clue2, clue3=clue3)
            db.session.add(new_hunt)
            db.session.commit()
            
            flash('Scavenger hunt created successfully!')
            return redirect(url_for('index'))
            
    return render_template('create.html')

@app.route('/play/<int:hunt_id>', methods=['GET', 'POST'])
def play_hunt(hunt_id):
    hunt = Hunt.query.get_or_404(hunt_id)
    
    if request.method == 'POST':
        if 'attempt_image' not in request.files:
            flash('No image uploaded.')
            return redirect(request.url)
            
        file = request.files['attempt_image']
        if file.filename == '' or not allowed_file(file.filename):
            flash('Invalid or missing file.')
            return redirect(request.url)
            
        clues_used = int(request.form.get('clues_used', 1))
        
        filename = secure_filename(file.filename)
        attempt_path = os.path.join(app.config['UPLOAD_FOLDER'], 'attempt_' + filename)
        file.save(attempt_path)
        
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], hunt.target_image)
        
        is_match = verify_match(target_path, attempt_path)
        score = calculate_score(clues_used, is_match)
        
        attempt = Attempt(hunt_id=hunt.id, attempt_image=filename, clues_used=clues_used, score=score, is_successful=is_match)
        db.session.add(attempt)
        db.session.commit()
        
        if is_match:
            flash(f'Success! You found it using {clues_used} clue(s). You scored {score} points!')
        else:
            flash('Not quite right. Keep searching!')
            
        return redirect(url_for('play_hunt', hunt_id=hunt.id))
        
    return render_template('play.html', hunt=hunt)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
"""

LAYOUT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Photo Scavenger Hunt</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

        :root {
            --bs-body-font-family: 'Roboto', sans-serif;
            /* Blue Color Scheme */
            --m3-surface: #F4F8FB;
            --m3-surface-container: #E3EFFF;
            --m3-on-surface: #1E293B;
            --m3-primary: #1976D2;
            --m3-primary-hover: #1565C0;
            --m3-secondary: #0288D1;
            --m3-success: #2E7D32;
            --m3-warning: #ED6C02;
            --m3-outline: #90CAF9;
        }

        body {
            background-color: var(--m3-surface);
            color: var(--m3-on-surface);
            font-family: var(--bs-body-font-family);
        }

        /* Material 3 App Bar */
        .navbar.bg-dark {
            background-color: var(--m3-primary) !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            padding-top: 12px;
            padding-bottom: 12px;
            margin-bottom: 24px;
            border-bottom: none;
        }
        .navbar-dark .navbar-brand, 
        .navbar-dark .nav-link {
            color: #ffffff !important;
            font-weight: 500;
        }

        /* Material 3 Expressive Cards */
        .card {
            border: 1px solid #dee2e6;
            border-radius: 16px;
            background-color: var(--m3-surface-container);
            box-shadow: none;
            overflow: hidden;
        }
        .card-header {
            background-color: transparent;
            border-bottom: 1px solid #dee2e6;
            font-size: 1.1rem;
            font-weight: 500;
            padding: 16px 24px;
        }
        .card-body {
            padding: 24px;
        }

        /* Material 3 Buttons */
        .btn {
            border-radius: 100px; /* Fully rounded pill shape */
            padding: 10px 24px;
            font-weight: 500;
            border: none;
            transition: all 0.2s ease;
        }
        .btn-primary { background-color: var(--m3-primary); color: #fff; }
        .btn-primary:hover { background-color: var(--m3-primary-hover); box-shadow: 0 2px 6px rgba(0,0,0,0.15); }
        .btn-success { background-color: #386A20; color: #fff; }
        .btn-warning { background-color: var(--m3-warning); color: #fff; }
        .btn-secondary { background-color: #625B71; color: #fff; }

        /* Material 3 Filled Inputs */
        .form-control {
            border-radius: 4px;
            padding: 16px;
            background-color: var(--m3-surface);
            border: 1px solid var(--m3-outline);
            box-shadow: none !important;
        }
        .form-control:focus {
            background-color: var(--m3-surface);
            border-color: var(--m3-primary);
            box-shadow: 0 0 0 2px var(--m3-primary, 0.25) !important;
        }
        .form-label {
            font-weight: 500;
            color: var(--m3-on-surface);
        }

        /* Expressive Lists */
        .list-group-item {
            border: 1px solid #dee2e6;
            margin-bottom: 8px;
            border-radius: 12px !important;
            background-color: var(--m3-surface-container);
        }
        .list-group-item.list-group-item-action:hover {
            background-color: #BBDEFB;
        }
        .list-group { background: transparent; }

        /* Flashed messages */
        .alert-info {
            background-color: var(--m3-surface-container);
            color: var(--m3-on-surface);
            border: 1px solid var(--m3-primary);
            border-radius: 12px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">Scavenger Hunt</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <div class="navbar-nav ms-auto">
                    <a class="nav-link" href="/create">Create Hunt</a>
                </div>
            </div>
        </div>
    </nav>
    <div class="container mt-4">
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="alert alert-info">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        {% block content %}{% endblock %}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

INDEX_HTML = r"""{% extends 'layout.html' %}

{% block content %}
<h2>Active Hunts</h2>
<div class="list-group mt-3">
    {% for hunt in hunts %}
        <a href="{{ url_for('play_hunt', hunt_id=hunt.id) }}" class="list-group-item list-group-item-action">
            Hunt #{{ hunt.id }} - Play Now!
        </a>
    {% else %}
        <p>No hunts available yet. Be the first to <a href="/create">create one</a>!</p>
    {% endfor %}
</div>
{% endblock %}
"""

CREATE_HTML = r"""{% extends 'layout.html' %}

{% block content %}
<h2>Create a New Scavenger Hunt</h2>
<form method="POST" enctype="multipart/form-data" class="mt-4">
    <div class="mb-3">
        <label for="target_image" class="form-label">Target Image (What players need to find)</label>
        <input type="file" class="form-control" id="target_image" name="target_image" accept="image/png, image/jpeg" required>
    </div>
    <div class="mb-3">
        <label for="clue1" class="form-label">Clue 1 (Hardest / Most Cryptic)</label>
        <input type="text" class="form-control" id="clue1" name="clue1" required>
    </div>
    <div class="mb-3">
        <label for="clue2" class="form-label">Clue 2 (Medium Helpfulness)</label>
        <input type="text" class="form-control" id="clue2" name="clue2" required>
    </div>
    <div class="mb-3">
        <label for="clue3" class="form-label">Clue 3 (Easiest / Most Direct)</label>
        <input type="text" class="form-control" id="clue3" name="clue3" required>
    </div>
    <button type="submit" class="btn btn-primary">Create Hunt</button>
</form>
{% endblock %}
"""

PLAY_HTML = r"""{% extends 'layout.html' %}

{% block content %}
<h2>Play Hunt #{{ hunt.id }}</h2>

<div class="card mt-4 mb-4">
    <div class="card-header">Clues</div>
    <div class="card-body">
        <p id="display-clue1" class="fw-bold">1. {{ hunt.clue1 }}</p>
        <p id="display-clue2" class="fw-bold text-muted" style="display: none;">2. {{ hunt.clue2 }}</p>
        <p id="display-clue3" class="fw-bold text-muted" style="display: none;">3. {{ hunt.clue3 }}</p>
        
        <button type="button" class="btn btn-warning" id="revealBtn" onclick="revealNextClue()">Reveal Next Clue</button>
        </div>
    </div>
</div>

<form method="POST" enctype="multipart/form-data">
    <input type="hidden" id="clues_used" name="clues_used" value="1">
    <div class="mb-3">
        <label for="attempt_image" class="form-label">Upload your photo attempt</label>
        <input type="file" class="form-control" id="attempt_image" name="attempt_image" accept="image/png, image/jpeg" required>
    </div>
    <div tton>
    </div>
</form>

<script>
    let cluesRevealed = 1;
    function revealNextClue() {
        if (cluesRevealed === 1) {
            document.getElementById('display-clue2').style.display = 'block';
            cluesRevealed = 2;
        } else if (cluesRevealed === 2) {
            document.getElementById('display-clue3').style.display = 'block';
            cluesRevealed = 3;
            document.getElementById('revealBtn').style.display = 'none';
        }
        document.getElementById('clues_used').value = cluesRevealed;
    }
</script>
{% endblock %}
"""

def create_project():
    base_dir = "."
    templates_dir = os.path.join(base_dir, "templates")
    uploads_dir = os.path.join(base_dir, "uploads")
    
    # Create directories
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Define file paths and their contents
    files = {
        os.path.join(base_dir, "app.py"): APP_PY,
        os.path.join(templates_dir, "layout.html"): LAYOUT_HTML,
        os.path.join(templates_dir, "index.html"): INDEX_HTML,
        os.path.join(templates_dir, "create.html"): CREATE_HTML,
        os.path.join(templates_dir, "play.html"): PLAY_HTML,
    }
    
    # Write files
    for filepath, content in files.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created: {filepath}")

    print("\nProject generated successfully!")
    print("\nTo start the app, run the following commands:")
    print("pip install Flask Flask-SQLAlchemy opencv-python numpy Werkzeug pillow google-genai")
    print("python app.py")

if __name__ == "__main__":
    create_project()
