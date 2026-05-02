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
if GEMINI_API_KEY and GEMINI_API_KEY.strip() and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None

class Hunt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    target_image = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
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

def verify_match(target_path, attempt_path, min_match_count=10, min_inliers=8, ratio_thresh=0.75):
    # Use Gemini API for smart, semantic image matching if an API key is provided
    if client:
        try:
            target_img = Image.open(target_path)
            attempt_img = Image.open(attempt_path)
            
            prompt = (
                "You are an AI judge for a photo scavenger hunt. Your task is to determine if a player's photo is a match for a target photo. "
                "The player's photo might have different lighting, angles, or zoom levels, but it must clearly show the same primary subject or location. "
                "Be reasonably flexible, but do not accept photos of completely different objects or places. "
                "Answer ONLY with 'True' if the main subject is a clear match, or 'False' if it is not."
            )
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model='gemini-1.5-flash',
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
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['target_image']
        if file.filename == '':
            flash('No selected file', 'warning')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            clue1 = request.form.get('clue1')
            clue2 = request.form.get('clue2')
            clue3 = request.form.get('clue3')
            description = request.form.get('description')
            
            new_hunt = Hunt(target_image=filename, description=description, clue1=clue1, clue2=clue2, clue3=clue3)
            db.session.add(new_hunt)
            db.session.commit()
            
            flash('Scavenger hunt created successfully!', 'success')
            return redirect(url_for('index'))
            
    return render_template('create.html')

@app.route('/play/<int:hunt_id>', methods=['GET', 'POST'])
def play_hunt(hunt_id):
    hunt = Hunt.query.get_or_404(hunt_id)
    
    if request.method == 'POST':
        if 'attempt_image' not in request.files:
            flash('No image uploaded.', 'danger')
            return redirect(request.url)
            
        file = request.files['attempt_image']
        if file.filename == '' or not allowed_file(file.filename):
            flash('Invalid or missing file.', 'danger')
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
            flash(f'Success! You found it using {clues_used} clue(s). You scored {score} points!', 'success')
        else:
            flash('Not quite right. Keep searching!', 'danger')
            
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
    <!-- FontAwesome for travel/search icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Nunito:wght@400;600;700&display=swap');

        :root {
            --bg-color: #F4EED3; /* Vintage map sand */
            --text-color: #2b2b2b;
            --primary-color: #3b5323; /* Forest green */
            --primary-hover: #2c3e1a;
            --secondary-color: #8b5a2b; /* Leather brown */
            --card-bg: #FCFAEF;
            --border-color: #D4C4A8;
        }

        body {
            background-color: var(--bg-color);
            background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23d4c4a8' fill-opacity='0.4'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
            color: var(--text-color);
            font-family: 'Nunito', sans-serif;
        }

        h1, h2, h3, h4, h5, h6, .navbar-brand {
            font-family: 'Playfair Display', serif;
            font-weight: 700;
        }

        /* Travel Theme App Bar */
        .navbar.bg-dark {
            background-color: var(--primary-color) !important;
            border-bottom: 4px solid var(--secondary-color);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding-top: 15px;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }
        .navbar-brand {
            font-size: 1.5rem;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        .navbar-dark .navbar-brand,
        .navbar-dark .nav-link {
            color: #ffffff !important;
        }
        .navbar-dark .nav-link {
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9rem;
            letter-spacing: 0.5px;
        }

        /* Postcard/Journal Cards */
        .card {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background-color: var(--card-bg);
            box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
            overflow: hidden;
            position: relative;
        }
        .card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 4px;
            background: repeating-linear-gradient(
                45deg,
                var(--secondary-color),
                var(--secondary-color) 10px,
                #fff 10px,
                #fff 20px,
                #d32f2f 20px,
                #d32f2f 30px,
                #fff 30px,
                #fff 40px
            ); /* Airmail edge effect */
        }
        .card-header {
            background-color: transparent;
            border-bottom: 1px dashed var(--border-color);
            font-size: 1.2rem;
            font-family: 'Playfair Display', serif;
            padding: 20px 24px 10px;
            color: var(--primary-color);
        }
        .card-body {
            padding: 24px;
        }

        /* Compass/Travel Buttons */
        .btn {
            border-radius: 4px;
            padding: 10px 20px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.85rem;
            border: none;
            transition: all 0.3s ease;
        }
        .btn-primary { background-color: var(--primary-color); color: #fff; }
        .btn-primary:hover { background-color: var(--primary-hover); box-shadow: 0 4px 8px rgba(0,0,0,0.2); transform: translateY(-1px); }
        .btn-success { background-color: var(--primary-color); color: #fff; border: 1px solid var(--primary-color); }
        .btn-warning { background-color: var(--secondary-color); color: #fff; }
        .btn-warning:hover { background-color: #6d4622; color: #fff; }
        .btn-secondary { background-color: #555; color: #fff; }
        .btn-outline-info { color: var(--secondary-color); border-color: var(--secondary-color); border-width: 2px; }
        .btn-outline-info:hover { background-color: var(--secondary-color); color: #fff; }

        /* Form Inputs */
        .form-control {
            border-radius: 4px;
            padding: 12px 16px;
            background-color: #fff;
            border: 1px solid var(--border-color);
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.05) !important;
        }
        .form-control:focus {
            border-color: var(--primary-color);
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.05), 0 0 0 2px rgba(59, 83, 35, 0.25) !important;
        }
        .form-label {
            font-weight: 600;
            color: var(--primary-color);
        }

        /* Expressive Lists */
        .list-group-item {
            border: 1px solid var(--border-color);
            margin-bottom: 12px;
            border-radius: 8px !important;
            background-color: var(--card-bg);
            box-shadow: 1px 1px 4px rgba(0,0,0,0.03);
        }
        .list-group-item.list-group-item-action:hover {
            background-color: #F0E8D5;
        }
        .list-group { background: transparent; }

        /* Flashed messages */
        .alert-info {
            background-color: #E8F0E5;
            color: var(--primary-color);
            border: 1px solid var(--primary-color);
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Badges */
        .badge.bg-primary { background-color: var(--secondary-color) !important; }
        .badge.bg-secondary { background-color: #6c757d !important; }
        .badge.bg-info { background-color: var(--primary-color) !important; color: white !important; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fa-solid fa-compass me-2"></i>Scavenger Hunt</a>
            <div class="navbar-nav ms-auto d-flex flex-row gap-3">
            </div>
        </div>
    </nav>
    <div class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    {% set alert_class = category if category in ['success', 'danger', 'warning', 'info'] else 'info' %}
                    <div class="alert alert-{{ alert_class }} shadow-sm alert-dismissible fade show" role="alert">
                        <i class="fa-solid fa-circle-info me-2"></i>{{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
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
<div class="row">
    <div class="col-md-7">
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h2><i class="fa-solid fa-earth-americas me-2"></i>Active Expeditions</h2>
            <div class="d-flex gap-2">
                <a href="/create" class="btn btn-primary"><i class="fa-solid fa-map-location-dot me-1"></i> Create Hunt</a>
                <button type="button" class="btn btn-outline-info" onclick="findHuntsNearMe()"><i class="fa-solid fa-location-crosshairs me-1"></i> Near Me</button>
            </div>
        </div>
        <p id="location-status" class="text-muted"></p>
        
        <div class="list-group">
            {% for hunt in hunts %}
                <div class="list-group-item d-flex flex-column flex-sm-row justify-content-between align-items-start align-items-sm-center gap-2 p-3">
                    <a href="{{ url_for('play_hunt_start', hunt_id=hunt.id) }}" class="text-decoration-none text-reset flex-grow-1 mb-2 mb-sm-0 fw-bold fs-5">
                        <i class="fa-solid fa-map me-2 text-muted"></i>{{ hunt.name }}
                    </a>
                    {% if hunt.min_distance is defined and hunt.min_distance is not none %}
                        <span class="badge bg-secondary rounded-pill me-2"><i class="fa-solid fa-ruler me-1"></i>{{ "%.1f"|format(hunt.min_distance) }} miles</span>
                    {% endif %}
                    {% if session.is_admin %}
                    <div class="admin-actions">
                        <a href="{{ url_for('edit_hunt', hunt_id=hunt.id) }}" class="btn btn-sm btn-secondary"><i class="fa-solid fa-pen"></i></a>
                        <form action="{{ url_for('delete_hunt', hunt_id=hunt.id) }}" method="POST" class="d-inline" onsubmit="return confirm('Delete this expedition?');">
                            <button type="submit" class="btn btn-sm btn-danger"><i class="fa-solid fa-trash"></i></button>
                        </form>
                    </div>
                    {% endif %}
                </div>
            {% else %}
                <p>No expeditions available yet. Be the first to <a href="/create">chart one</a>!</p>
            {% endfor %}
        </div>
    </div>

    <div class="col-md-5 mt-5 mt-md-0">
        <h2><i class="fa-solid fa-trophy me-2"></i>Explorer Leaderboard</h2>
        <div class="card mt-3">
            <ul class="list-group list-group-flush">
                {% for row in leaderboard %}
                    <li class="list-group-item d-flex justify-content-between align-items-center p-3">
                        <a href="{{ url_for('user_details', username=row.username) }}" class="fw-bold text-decoration-none text-dark"><i class="fa-solid fa-user-astronaut me-2"></i>{{ row.username }}</a>
                        <div class="d-flex align-items-center gap-2">
                            <span class="badge bg-primary rounded-pill">{{ row.total }} pts</span>
                            {% if session.is_admin %}
                            <form action="{{ url_for('delete_player', username=row.username) }}" method="POST" class="m-0" onsubmit="return confirm('Delete this explorer?');">
                                <button type="submit" class="btn btn-sm btn-danger"><i class="fa-solid fa-xmark"></i></button>
                            </form>
                            {% endif %}
                        </div>
                    </li>
                {% else %}
                    <li class="list-group-item text-muted">No completed hunts yet.</li>
                {% endfor %}
            </ul>
        </div>
    </div>
</div>

<script>
function findHuntsNearMe() {
    const status = document.getElementById('location-status');
    status.textContent = 'Getting your location...';
    
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                window.location.href = `/?lat=${lat}&lon=${lon}`;
            },
            function(error) {
                status.textContent = 'Error getting location: ' + error.message;
                status.className = 'text-danger';
            }
        );
    } else {
        status.textContent = 'Geolocation is not supported by your browser.';
        status.className = 'text-danger';
    }
}
</script>
{% endblock %}
"""

CREATE_HTML = r"""{% extends 'layout.html' %}

{% block content %}
<h2><i class="fa-solid fa-map-marked-alt me-2"></i>Chart a New Expedition</h2>
<div class="card mt-4 p-2">
    <div class="card-body">
        <form method="POST" enctype="multipart/form-data">
            <div class="mb-3">
                <label for="hunt_name" class="form-label"><i class="fa-solid fa-signature me-1"></i> Expedition Name</label>
                <input type="text" class="form-control" id="hunt_name" name="hunt_name" placeholder="E.g., The Lost City of Gold" required>
            </div>

            <div class="mb-3">
                <label for="target_image" class="form-label"><i class="fa-solid fa-image me-1"></i> Target Image</label>
                <input type="file" class="form-control" id="target_image" name="target_image" accept="image/png, image/jpeg" required>
            </div>
            <div class="mb-3">
                <label for="description" class="form-label"><i class="fa-solid fa-book-open me-1"></i> Field Notes (Optional)</label>
                <textarea class="form-control" id="description" name="description" rows="2" placeholder="Brief description of the photo or location..."></textarea>
            </div>
            <div class="mb-3">
                <label for="clue1" class="form-label"><i class="fa-solid fa-magnifying-glass me-1"></i> Clue 1 (Cryptic)</label>
                <input type="text" class="form-control" id="clue1" name="clue1" required>
            </div>
            <div class="mb-3">
                <label for="clue2" class="form-label"><i class="fa-solid fa-magnifying-glass-plus me-1"></i> Clue 2 (Moderate)</label>
                <input type="text" class="form-control" id="clue2" name="clue2" required>
            </div>
            <div class="mb-3">
                <label for="clue3" class="form-label"><i class="fa-solid fa-magnifying-glass-location me-1"></i> Clue 3 (Direct)</label>
                <input type="text" class="form-control" id="clue3" name="clue3" required>
            </div>
            
            <div class="mb-4">
                <label class="form-label"><i class="fa-solid fa-location-dot me-1"></i> Coordinates (Optional)</label>
                <div class="input-group mb-2">
                    <input type="text" class="form-control address-input" name="address" placeholder="Enter an address or landmark" onblur="geocodeAddress(this)">
                    <button type="button" class="btn btn-outline-info" onclick="geocodeAddress(this)"><i class="fa-solid fa-search"></i> Search</button>
                </div>
                <input type="hidden" name="latitude" class="lat-input">
                <input type="hidden" name="longitude" class="lng-input">
                <button type="button" class="btn btn-outline-secondary btn-sm" onclick="getLocation(this)"><i class="fa-solid fa-thumbtack me-1"></i> Drop Pin Here</button>
                <span class="location-status text-muted ms-2 small"></span>
            </div>
            
            <button type="submit" class="btn btn-primary btn-lg w-100"><i class="fa-solid fa-paper-plane me-2"></i> Launch Expedition</button>
        </form>
    </div>
</div>

<script>
function geocodeAddress(element) {
    const container = element.closest('.mb-3');
    const addressInput = container.querySelector('.address-input');
    const latInput = container.querySelector('.lat-input');
    const lngInput = container.querySelector('.lng-input');
    const statusSpan = container.querySelector('.location-status');
    const address = addressInput.value.trim();

    if (!address) return;

    statusSpan.textContent = "Finding coordinates...";
    statusSpan.className = "location-status text-warning ms-2 small fw-bold";

    fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(address)}`)
        .then(response => response.json())
        .then(data => {
            if (data && data.length > 0) {
                latInput.value = data[0].lat;
                lngInput.value = data[0].lon;
                statusSpan.textContent = "Coordinates found!";
                statusSpan.className = "location-status text-success ms-2 small fw-bold";
            } else {
                statusSpan.textContent = "Address not found.";
                statusSpan.className = "location-status text-danger ms-2 small fw-bold";
            }
        })
        .catch(error => {
            statusSpan.textContent = "Error finding address.";
            statusSpan.className = "location-status text-danger ms-2 small fw-bold";
        });
}

function getLocation(btn) {
    const container = btn.closest('.mb-3');
    const statusSpan = container.querySelector('.location-status');
    const latInput = container.querySelector('.lat-input');
    const lngInput = container.querySelector('.lng-input');
    const addressInput = container.querySelector('.address-input');

    if (navigator.geolocation) {
        statusSpan.textContent = "Getting location...";
        statusSpan.className = "location-status text-warning ms-2 fw-bold";
        
        navigator.geolocation.getCurrentPosition(
            function(position) {
                latInput.value = position.coords.latitude;
                lngInput.value = position.coords.longitude;
                
                fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${position.coords.latitude}&lon=${position.coords.longitude}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data && data.display_name) {
                            addressInput.value = data.display_name;
                        }
                    })
                    .catch(err => console.log(err));
                
                statusSpan.textContent = "Location pinned!";
                statusSpan.className = "location-status text-success ms-2 fw-bold";
            },
            function(error) {
                statusSpan.textContent = "Error: " + error.message;
                statusSpan.className = "location-status text-danger ms-2 fw-bold";
            }
        );
    } else {
        statusSpan.textContent = "Geolocation is not supported by this browser.";
        statusSpan.className = "location-status text-danger ms-2 fw-bold";
    }
}
</script>
{% endblock %}
"""

PLAY_HTML = r"""{% extends 'layout.html' %}

{% block content %}
<div class="d-flex justify-content-between align-items-center mb-4">
    <h2><i class="fa-solid fa-route me-2"></i>Expedition: {{ hunt.name }}</h2>
    <h4><span class="badge bg-info shadow-sm"><i class="fa-solid fa-star me-1"></i>Score: {{ player.total_score }}</span></h4>
</div>

<div class="card mt-4 mb-4">
    <div class="card-header"><i class="fa-solid fa-camera-retro me-2"></i>Target Objective</div>
    <div class="card-body text-center">
        <img src="{{ url_for('uploaded_file', filename=target.target_image) }}" class="img-fluid rounded border border-2 border-secondary p-1" alt="Target Image" style="max-height: 400px; background-color: #fff;">
        {% if target.description %}
        <p class="mt-4 mb-0 text-dark font-monospace bg-light p-3 rounded text-start border"><i class="fa-solid fa-quote-left me-2 text-muted"></i>{{ target.description }}</p>
        {% endif %}
    </div>
</div>

<div class="card mt-4 mb-4">
    <div class="card-header"><i class="fa-solid fa-scroll me-2"></i>Explorer's Log (Clues)</div>
    <div class="card-body">
        <p id="display-clue1" class="fw-bold fs-5"><i class="fa-solid fa-key me-2 text-warning"></i>1. {{ target.clue1 }}</p>
        <p id="display-clue2" class="fw-bold text-muted fs-5" style="display: none;"><i class="fa-solid fa-key me-2 text-warning"></i>2. {{ target.clue2 }}</p>
        <p id="display-clue3" class="fw-bold text-muted fs-5" style="display: none;"><i class="fa-solid fa-key me-2 text-warning"></i>3. {{ target.clue3 }}</p>
        
        <div class="d-grid gap-2 d-md-block mt-4">
            <button type="button" class="btn btn-warning" id="revealBtn" onclick="revealNextClue()"><i class="fa-solid fa-eye me-1"></i> Reveal Next Clue</button>
        </div>
    </div>
</div>

<form method="POST" enctype="multipart/form-data" class="mb-5 bg-white p-4 border rounded shadow-sm" id="attemptForm">
    <h4 class="mb-3"><i class="fa-solid fa-upload me-2"></i>Submit Your Findings</h4>
    <input type="hidden" id="clues_used" name="clues_used" value="1">
    <!-- This input is hidden from the user -->
    <input type="file" id="attempt_image" name="attempt_image" accept="image/*" style="display: none;">

    <div class="mb-4">
        <div class="d-grid gap-2 d-md-flex">
            <button type="button" class="btn btn-outline-secondary" id="uploadBtn"><i class="fa-solid fa-folder-open me-1"></i> From Library</button>
            <button type="button" class="btn btn-outline-info" id="captureBtn"><i class="fa-solid fa-camera me-1"></i> Take Photo</button>
        </div>
        <p id="file-info" class="text-muted mt-3 mb-0 font-monospace small"><i class="fa-solid fa-file-image me-1"></i> No image selected.</p>
    </div>
    <div class="d-grid gap-2 d-md-block">
        <button type="submit" class="btn btn-success btn-lg" id="submitBtn"><i class="fa-solid fa-check-circle me-1"></i> Verify Match</button>
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

    const fileInput = document.getElementById('attempt_image');
    const uploadBtn = document.getElementById('uploadBtn');
    const captureBtn = document.getElementById('captureBtn');
    const fileInfo = document.getElementById('file-info');
    const attemptForm = document.getElementById('attemptForm');
    const submitBtn = document.getElementById('submitBtn');

    uploadBtn.addEventListener('click', () => {
        fileInput.removeAttribute('capture');
        fileInput.click();
    });

    captureBtn.addEventListener('click', () => {
        fileInput.setAttribute('capture', 'environment'); // 'environment' for back camera
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileInfo.textContent = `Selected: ${fileInput.files[0].name}`;
            fileInfo.classList.remove('text-muted');
            fileInfo.classList.add('text-success', 'fw-bold');
        }
    });

    attemptForm.addEventListener('submit', function(e) {
        if (fileInput.files.length === 0) {
            e.preventDefault();
            alert("Please select or capture a photo first!");
            return;
        }
        submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-1"></i> Analyzing Photo...';
        submitBtn.disabled = true;
    });
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
