import os
import re

VIEWS_FILE = r"D:\4th_sem_project\implementation\facial_attendance_11\facial_attendance\homepage\views.py"

with open(VIEWS_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add new imports if missing
new_imports = """
import base64
import json
import numpy as np
import cv2
from django.core.files.base import ContentFile
"""
if "from django.core.files.base import ContentFile" not in content:
    content = content.replace("import base64\nimport json\n", new_imports + "\n")

# 2. Add global classifier instances
global_classifiers = """
# --- Global Classifiers ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_current_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists('trainer/trainer.yml'):
        try:
            recognizer.read('trainer/trainer.yml')
        except:
            pass
    return recognizer

CLOSED_FRAMES_MIN = 2
"""
content = re.sub(r'video_camera_instance = None\n', global_classifiers + '\n', content)

# 3. Replace video_feed, gen, stop_camera with new process_client_frame
new_frame_processing = """
@csrf_exempt
def process_client_frame(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)
        
    try:
        data = json.loads(request.body)
        image_data = data.get('image')
        if not image_data:
            return JsonResponse({'error': 'No image provided'}, status=400)
            
        format, imgstr = image_data.split(';base64,') 
        img_bytes = base64.b64decode(imgstr)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        frame = cv2.flip(frame, 1) # Mirror
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Session state initialization
        if 'eye_closed_frames' not in request.session:
            request.session['eye_closed_frames'] = 0
            request.session['blink_detected'] = False
            request.session['recognition_count'] = 0
            request.session['last_recognized_id'] = None
            
        eye_closed_frames = request.session['eye_closed_frames']
        blink_detected = request.session['blink_detected']
        recognition_count = request.session['recognition_count']
        last_recognized_id = request.session['last_recognized_id']
        
        recognizer = get_current_recognizer()
        
        drawing_commands = []
        current_recognized_id = None
        
        for (x, y, fw, fh) in faces:
            # Blink Detection
            roi_gray = gray[y: y + fh // 2, x: x + fw]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            
            if len(eyes) == 0:
                eye_closed_frames += 1
            else:
                if eye_closed_frames >= CLOSED_FRAMES_MIN:
                    blink_detected = True
                eye_closed_frames = 0
                
            # Face Recognition
            try:
                user_id, confidence = recognizer.predict(gray[y:y+fh, x:x+fw])
                if confidence < 75:
                    from .models import UserProfile
                    try:
                        profile = UserProfile.objects.get(user_id=user_id)
                        role = profile.role.capitalize()
                        name = profile.user.username
                        label = f"{name} ({role})"
                        
                        if profile.role == 'teacher':
                            color = [255, 0, 0] # BGR in OpenCV, but we use RGB for frontend overlay
                            rgb_color = 'rgb(0, 0, 255)' 
                        elif profile.role == 'admin':
                            rgb_color = 'rgb(255, 165, 0)'
                        else:
                            rgb_color = 'rgb(0, 255, 0)'
                            
                        current_recognized_id = user_id
                    except:
                        label = "Unknown Profile"
                        rgb_color = 'rgb(255, 0, 0)'
                else:
                    label = "Unknown"
                    rgb_color = 'rgb(255, 0, 0)'
            except Exception:
                label = "Scanning..."
                rgb_color = 'rgb(255, 165, 0)'
                
            drawing_commands.append({
                'type': 'rect',
                'x': int(x), 'y': int(y), 'w': int(fw), 'h': int(fh),
                'color': rgb_color,
                'label': label
            })
            
        # Update recognition counter
        if current_recognized_id and blink_detected:
            if current_recognized_id == last_recognized_id:
                recognition_count += 1
            else:
                last_recognized_id = current_recognized_id
                recognition_count = 1
        elif not blink_detected:
            recognition_count = 0
            last_recognized_id = None
            
        # Save state to session
        request.session['eye_closed_frames'] = eye_closed_frames
        request.session['blink_detected'] = blink_detected
        request.session['recognition_count'] = recognition_count
        request.session['last_recognized_id'] = last_recognized_id
        
        return JsonResponse({
            'status': 'success',
            'draw': drawing_commands,
            'liveness': blink_detected,
            'recognition_count': recognition_count
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def stop_camera(request):
    if 'blink_detected' in request.session:
        # Reset session vars when camera is stopped by user
        request.session['blink_detected'] = False
        request.session['recognition_count'] = 0
        request.session['last_recognized_id'] = None
    return JsonResponse({"status": "Session cleared"})

"""

# Regex replacement for the camera streaming helpers
content = re.sub(
    r'# --- Camera Streaming Helpers ---.*?(?=# --- Authentication Views ---)',
    new_frame_processing,
    content,
    flags=re.DOTALL
)

# 4. Fix verify_teacher_face
new_verify = """
def verify_teacher_face(request):
    pending_id = request.session.get('pending_2fa_user_id')
    if not pending_id:
        return JsonResponse({'status': 'error', 'message': 'No pending login'})

    blink_detected = request.session.get('blink_detected', False)
    last_recognized_id = request.session.get('last_recognized_id')
    recognition_count = request.session.get('recognition_count', 0)
    
    if not blink_detected:
        return JsonResponse({'status': 'pending', 'message': 'Please blink to verify liveness'})
        
    if not (last_recognized_id and recognition_count >= 10):
        return JsonResponse({'status': 'pending'})
        
    if last_recognized_id == pending_id:
        try:
            user = User.objects.get(id=pending_id)
            login(request, user)
            del request.session['pending_2fa_user_id']
            request.session['blink_detected'] = False
            request.session['recognition_count'] = 0
            request.session['last_recognized_id'] = None
            return JsonResponse({'status': 'success', 'url': '/dashboard/'})
        except User.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'User not found'})
            
    return JsonResponse({'status': 'pending'})
"""
content = re.sub(
    r'def verify_teacher_face\(request\):.*?return JsonResponse\(\{\'status\': \'pending\'\}\)',
    new_verify.strip(),
    content,
    flags=re.DOTALL
)

# 5. Add capture_enrollment_frame_api 
# We'll put it right before capture_face_samples
enroll_api = """
@csrf_exempt
def capture_enrollment_frame_api(request):
    user_id = request.session.get('enroll_user_id')
    if not user_id:
        return JsonResponse({'status': 'error', 'message': 'No user in enrollment queue'})
        
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)
        
    try:
        data = json.loads(request.body)
        image_data = data.get('image')
        if not image_data:
            return JsonResponse({'error': 'No image provided'}, status=400)
            
        # Initialize progress in session
        if 'capture_progress' not in request.session:
            request.session['capture_progress'] = 0
            
        progress = request.session['capture_progress']
        
        if progress >= 50:
             return JsonResponse({'status': 'complete', 'progress': 50})
             
        format, imgstr = image_data.split(';base64,') 
        img_bytes = base64.b64decode(imgstr)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if not os.path.exists('datasets'): os.makedirs('datasets')
        
        for (x, y, w, h) in faces:
            progress += 1
            cv2.imwrite(f"datasets/User.{user_id}.{progress}.jpg", gray[y:y+h, x:x+w])
            request.session['capture_progress'] = progress
            break # only 1 face per frame to avoid jumping progress
            
        return JsonResponse({'status': 'capturing', 'progress': progress})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

"""
content = content.replace("def capture_face_samples(request, user_id):", enroll_api + "\ndef capture_face_samples(request, user_id):")

# 6. Replace capture_progress_api logic
new_capture_progress = """
def capture_progress_api(request):
    progress = request.session.get('capture_progress', 0)
    return JsonResponse({'progress': progress})
"""
content = re.sub(
    r'def capture_progress_api\(request\):.*?return JsonResponse\(\{\'progress\': progress\}\)',
    new_capture_progress.strip(),
    content,
    flags=re.DOTALL
)

# 7. Replace save_face_data logic
new_save = """
@login_required
def save_face_data(request):
    user_id = request.session.get('enroll_user_id')
    if not user_id:
        return JsonResponse({'status': 'error', 'message': 'User ID missing'})

    # Retrain the model
    if run_trainer_logic():
        try:
            profile = UserProfile.objects.get(user_id=user_id)
            profile.is_face_registered = True
            profile.save()
            
            # Clear session
            del request.session['enroll_user_id']
            if 'capture_progress' in request.session:
                del request.session['capture_progress']
            
            return JsonResponse({'status': 'success'})
        except UserProfile.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Profile not found'})
            
    return JsonResponse({'status': 'error', 'message': 'Training failed'})
"""
content = re.sub(
    r'@login_required\ndef save_face_data\(request\):.*?return JsonResponse\(\{\'status\': \'error\', \'message\': \'Training failed\'\}\)',
    new_save.strip(),
    content,
    flags=re.DOTALL
)

# 8. Fix reset_recognition_session
new_reset = """
@csrf_exempt
def reset_recognition_session(request):
    request.session['last_recognized_id'] = None
    request.session['recognition_count'] = 0
    return JsonResponse({'status': 'ok'})
"""
content = re.sub(
    r'@csrf_exempt\ndef reset_recognition_session\(request\):.*?return JsonResponse\(\{\'status\': \'ok\'\}\)',
    new_reset.strip(),
    content,
    flags=re.DOTALL
)

# 9. Fix check_face_status
new_check = """
def check_face_status(request):
    last_recognized_id = request.session.get('last_recognized_id')
    recognition_count = request.session.get('recognition_count', 0)
    
    if not (last_recognized_id and recognition_count >= 10):
        return JsonResponse({'status': 'pending'})
        
    try:
        from .models import UserProfile, Student, Attendance
        from datetime import date, datetime
        from django.contrib.auth import login
        from django.contrib.auth.models import User
        
        user = User.objects.get(id=last_recognized_id)
        profile = UserProfile.objects.get(user=user)
        
        name = user.username
        role = profile.role.capitalize()
        roll_no = "FACULTY"
"""
content = re.sub(
    r'def check_face_status\(request\):.*?roll_no = "FACULTY"',
    new_check.strip(),
    content,
    flags=re.DOTALL
)

# also need to remove the global clear in check_face_status
content = content.replace("video_camera_instance.last_recognized_id = None\n            video_camera_instance.recognition_count = 0", "request.session['last_recognized_id'] = None\n            request.session['recognition_count'] = 0")

with open(VIEWS_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print("Views updated successfully.")
