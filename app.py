import os
import cv2
import numpy as np
import time
import math
from PIL import Image
from google import genai
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from sqlalchemy import func
from functools import wraps

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
    name = db.Column(db.String(255), nullable=False)
    targets = db.relationship('Target', backref='hunt', lazy=True, cascade="all, delete-orphan")

class Target(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hunt_id = db.Column(db.Integer, db.ForeignKey('hunt.id'), nullable=False)
    step_number = db.Column(db.Integer, nullable=False)
    target_image = db.Column(db.String(255), nullable=False)
    clue1 = db.Column(db.String(255), nullable=False)
    clue2 = db.Column(db.String(255), nullable=False)
    clue3 = db.Column(db.String(255), nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    attempts = db.relationship('Attempt', backref='target', lazy=True)

class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False)
    hunt_id = db.Column(db.Integer, db.ForeignKey('hunt.id'), nullable=False)
    total_score = db.Column(db.Integer, default=0)
    is_completed = db.Column(db.Boolean, default=False)
    attempts = db.relationship('Attempt', backref='player', lazy=True)

class Attempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    target_id = db.Column(db.Integer, db.ForeignKey('target.id'), nullable=False)
    player_id = db.Column(db.Integer, db.ForeignKey('player.id'), nullable=False)
    attempt_image = db.Column(db.String(255), nullable=False)
    clues_used = db.Column(db.Integer, nullable=False)
    score = db.Column(db.Integer, nullable=False)
    is_successful = db.Column(db.Boolean, nullable=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_score(clues_used, is_successful):
    if not is_successful: return -10
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

        # Resize images to a standard maximum dimension.
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

        # Use BFMatcher with knnMatch and Lowe's ratio test for more robust matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test to find good matches
        good_matches = []
        for match_pair in matches:
            # Ensure there are two matches to compare
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        
        print(f"DEBUG: Found {len(good_matches)} good matches before geometric check.")
        
        if len(good_matches) >= min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography to enforce geometric consistency (RANSAC)
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

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin'):
            flash('Admin access required.', 'danger')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve a file from the upload folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def haversine(lat1, lon1, lat2, lon2):
    R = 3959.87433 # Earth radius in miles
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dLon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

@app.route('/')
def index():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    hunts = Hunt.query.all()
    
    for hunt in hunts:
        hunt.min_distance = None
        
    if lat is not None and lon is not None:
        for hunt in hunts:
            min_dist = float('inf')
            for target in hunt.targets:
                if target.latitude is not None and target.longitude is not None:
                    dist = haversine(lat, lon, target.latitude, target.longitude)
                    if dist < min_dist:
                        min_dist = dist
            hunt.min_distance = min_dist if min_dist != float('inf') else None
            
        hunts.sort(key=lambda x: x.min_distance if getattr(x, 'min_distance', None) is not None else float('inf'))

    leaderboard = db.session.query(
        Player.username,
        func.sum(Player.total_score).label('total')
    ).filter_by(is_completed=True).group_by(Player.username).order_by(func.sum(Player.total_score).desc()).limit(10).all()
    
    return render_template('index.html', hunts=hunts, leaderboard=leaderboard)

@app.route('/create', methods=['GET', 'POST'])
def create_hunt():
    if request.method == 'POST':
        hunt_name = request.form.get('hunt_name', 'Untitled Hunt')
        new_hunt = Hunt(name=hunt_name)
        db.session.add(new_hunt)
        db.session.flush() # get new_hunt.id to associate targets
        
        target_images = request.files.getlist('target_image')
        clue1s = request.form.getlist('clue1')
        clue2s = request.form.getlist('clue2')
        clue3s = request.form.getlist('clue3')
        latitudes = request.form.getlist('latitude')
        longitudes = request.form.getlist('longitude')
        
        for i, file in enumerate(target_images):
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filename = f"hunt{new_hunt.id}_step{i+1}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                lat = float(latitudes[i]) if i < len(latitudes) and latitudes[i] else None
                lng = float(longitudes[i]) if i < len(longitudes) and longitudes[i] else None
                
                target = Target(hunt_id=new_hunt.id, step_number=i+1, target_image=filename,
                                clue1=clue1s[i] if i < len(clue1s) else '',
                                clue2=clue2s[i] if i < len(clue2s) else '',
                                clue3=clue3s[i] if i < len(clue3s) else '',
                                latitude=lat, longitude=lng)
                db.session.add(target)
        
        db.session.commit()
        flash('Scavenger hunt created successfully!')
        return redirect(url_for('index'))
            
    return render_template('create.html')

@app.route('/hunt/<int:hunt_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_hunt(hunt_id):
    hunt = Hunt.query.get_or_404(hunt_id)
    if request.method == 'POST':
        hunt.name = request.form.get('hunt_name', 'Untitled Hunt')
        
        # For simplicity, we delete old targets and their images, then recreate them.
        for target in hunt.targets:
            Attempt.query.filter_by(target_id=target.id).delete()
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], target.target_image))
            except OSError as e:
                print(f"Error deleting old file {target.target_image}: {e}")
        Target.query.filter_by(hunt_id=hunt_id).delete()

        target_images = request.files.getlist('target_image')
        clue1s = request.form.getlist('clue1')
        clue2s = request.form.getlist('clue2')
        clue3s = request.form.getlist('clue3')
        latitudes = request.form.getlist('latitude')
        longitudes = request.form.getlist('longitude')
        
        for i, file in enumerate(target_images):
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filename = f"hunt{hunt.id}_step{i+1}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                lat = float(latitudes[i]) if i < len(latitudes) and latitudes[i] else None
                lng = float(longitudes[i]) if i < len(longitudes) and longitudes[i] else None
                
                target = Target(hunt_id=hunt.id, step_number=i+1, target_image=filename,
                                clue1=clue1s[i], clue2=clue2s[i], clue3=clue3s[i],
                                latitude=lat, longitude=lng)
                db.session.add(target)
        
        db.session.commit()
        flash(f'Hunt "{hunt.name}" updated successfully.')
        return redirect(url_for('index'))

    return render_template('edit_hunt.html', hunt=hunt)

@app.route('/hunt/<int:hunt_id>/delete', methods=['POST'])
@admin_required
def delete_hunt(hunt_id):
    hunt = Hunt.query.get_or_404(hunt_id)
    hunt_name = hunt.name
    # Clean up image files
    for target in hunt.targets:
        Attempt.query.filter_by(target_id=target.id).delete()
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], target.target_image))
        except OSError as e:
            print(f"Error deleting file {target.target_image}: {e}")
            
    Player.query.filter_by(hunt_id=hunt.id).delete()
    db.session.delete(hunt)
    db.session.commit()
    flash(f'Hunt "{hunt_name}" has been deleted.')
    return redirect(url_for('index'))

@app.route('/player/<username>/delete', methods=['POST'])
@admin_required
def delete_player(username):
    player = Player.query.filter_by(username=username).first()
    if player:
        Attempt.query.filter_by(player_id=player.id).delete()
        db.session.delete(player)
        db.session.commit()
        flash(f'Player "{username}" and all their records have been deleted.')
    return redirect(url_for('index'))

@app.route('/user/<username>')
def user_details(username):
    # Query for all completed hunts by this user, joining with the Hunt table to get the hunt name.
    user_scores = db.session.query(
        Hunt.name,
        Player.total_score
    ).join(Player, Hunt.id == Player.hunt_id)\
     .filter(Player.username == username, Player.is_completed == True)\
     .order_by(Hunt.name)\
     .all()

    total_score_all_time = sum(score for _, score in user_scores)
    return render_template('user_details.html', username=username, scores=user_scores, total_score=total_score_all_time)

@app.route('/play/<int:hunt_id>', methods=['GET', 'POST'])
def play_hunt_start(hunt_id):
    hunt = Hunt.query.get_or_404(hunt_id)
    
    if request.method == 'POST':
        username = request.form.get('username')
        if not username:
            flash('Please enter a username.')
            return redirect(request.url)
            
        player = Player(username=username, hunt_id=hunt.id)
        db.session.add(player)
        db.session.commit()
        
        session[f'player_{hunt_id}'] = player.id
        return redirect(url_for('play_hunt_step', hunt_id=hunt.id, step_number=1))
        
    return render_template('start.html', hunt=hunt)

@app.route('/play/<int:hunt_id>/<int:step_number>', methods=['GET', 'POST'])
def play_hunt_step(hunt_id, step_number):
    hunt = Hunt.query.get_or_404(hunt_id)
    player_id = session.get(f'player_{hunt_id}')
    
    if not player_id:
        flash('Please start the hunt by entering your username.')
        return redirect(url_for('play_hunt_start', hunt_id=hunt.id))
        
    player = Player.query.get(player_id)
    target = Target.query.filter_by(hunt_id=hunt_id, step_number=step_number).first()
    
    if not target:
        if not player.is_completed:
            player.is_completed = True
            db.session.commit()
        flash(f'Congratulations {player.username}! You have completed the scavenger hunt with a total score of {player.total_score}!')
        return redirect(url_for('index'))
    
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
        attempt_path = os.path.join(app.config['UPLOAD_FOLDER'], f'attempt_{target.id}_' + filename)
        file.save(attempt_path)
        
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], target.target_image)
        
        is_match = verify_match(target_path, attempt_path)
        score = calculate_score(clues_used, is_match)
        
        attempt = Attempt(target_id=target.id, player_id=player.id, attempt_image=filename, clues_used=clues_used, score=score, is_successful=is_match)
        db.session.add(attempt)
        
        player.total_score += score
        db.session.commit()
        
        if is_match:
            flash(f'Success! You found it using {clues_used} clue(s). You scored {score} points!')
            return redirect(url_for('play_hunt_step', hunt_id=hunt.id, step_number=step_number + 1))
        else:
            flash('Not quite right. That cost you 10 points! Keep searching.')
            return redirect(request.url)
        
    return render_template('play.html', hunt=hunt, target=target, player=player)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('is_admin'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        if request.form.get('username') == 'admin' and request.form.get('password') == 'admin':
            session['is_admin'] = True
            flash('Logged in as admin.')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials.', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    flash('You have been logged out.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
