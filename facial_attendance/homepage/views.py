import os, cv2, time, math, base64, json, random
import numpy as np
from datetime import date, datetime
from PIL import Image

from django.shortcuts import render, redirect, get_object_or_404
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.contrib import messages
from django.contrib.auth import logout, login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
from django.conf import settings
from django.core.files.base import ContentFile
import csv

from .models import UserProfile, Student, Attendance, Teacher, AppSetting
from .forms import StudentRegistrationForm
from .camera import VideoCamera, get_detector, get_recognizer

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ─────────────────────────────────────────────
#  CACHED RECOGNIZER  (reload only when trainer.yml changes)
# ─────────────────────────────────────────────

_recognizer_cache = None
_trainer_mtime    = None

def get_current_recognizer():
    global _recognizer_cache, _trainer_mtime

    if not os.path.exists(TRAINER_FILE):
        return None  # No model trained yet

    current_mtime = os.path.getmtime(TRAINER_FILE)

    # Only reload if file changed on disk (e.g. after new enrollment)
    if _recognizer_cache is None or current_mtime != _trainer_mtime:
        rec = cv2.face.LBPHFaceRecognizer_create()
        try:
            rec.read(TRAINER_FILE)
            _recognizer_cache = rec
            _trainer_mtime    = current_mtime
            print("[INFO] Recognizer reloaded from disk.")
        except Exception as e:
            print(f"[ERROR] Could not load trainer.yml: {e}")
            return None

    return _recognizer_cache


# ─────────────────────────────────────────────
#  GLOBAL MODELS  (loaded once at startup)
# ─────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Face Landmarker  (blink + pose + smile)
_face_base   = python.BaseOptions(model_asset_path=os.path.join(settings.BASE_DIR, 'face_landmarker.task'))
_face_opts   = vision.FaceLandmarkerOptions(
    base_options=_face_base,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(_face_opts)

# Hand Landmarker  (show-hand challenge)
_hand_base   = python.BaseOptions(model_asset_path=os.path.join(settings.BASE_DIR, 'hand_landmarker.task'))
_hand_opts   = vision.HandLandmarkerOptions(
    base_options=_hand_base,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_landmarker = vision.HandLandmarker.create_from_options(_hand_opts)


# ─────────────────────────────────────────────
#  LANDMARK INDEX CONSTANTS
# ─────────────────────────────────────────────

# Eyes  (EAR blink)
LEFT_EYE_INDICES  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Pose  (head turn)
NOSE_TIP     = 1
LEFT_CHEEK   = 234    # outer left
RIGHT_CHEEK  = 454    # outer right

# Smile  (mouth corners vs mouth height)
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14

# Hand  (open-palm: all 5 fingertips above their MCP joint)
FINGERTIP_IDS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
FINGER_MCP_IDS = [2, 5, 9, 13, 17]  # corresponding base joints

# ─────────────────────────────────────────────
#  CHALLENGE POOL  — add / remove freely
# ─────────────────────────────────────────────

CHALLENGES = [
    {
        'key':         'blink',
        'instruction': 'Please BLINK your eyes',
        'emoji':       '👁️',
    },
    {
        'key':         'turn_left',
        'instruction': 'Turn your head to the LEFT',
        'emoji':       '⬅️',
    },
    {
        'key':         'turn_right',
        'instruction': 'Turn your head to the RIGHT',
        'emoji':       '➡️',
    },
    {
        'key':         'smile',
        'instruction': 'Give a big SMILE 😄',
        'emoji':       '😄',
    },
    {
        'key':         'show_hand',
        'instruction': 'Show an OPEN HAND to the camera ✋',
        'emoji':       '✋',
    },
]


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def eye_aspect_ratio(eye_pts):
    v1 = math.dist(eye_pts[1], eye_pts[5])
    v2 = math.dist(eye_pts[2], eye_pts[4])
    h  = math.dist(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0


def check_blink(landmarks, frame_shape, session):
    """Returns True once a valid blink is detected using a state machine."""
    h, w, _ = frame_shape
    lep = [(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE_INDICES]
    rep = [(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE_INDICES]
    ear = (eye_aspect_ratio(lep) + eye_aspect_ratio(rep)) / 2.0

    state  = session.get('ch_blink_state', 'OPEN')
    frames = session.get('ch_closed_frames', 0)

    EAR_CLOSE = 0.22
    EAR_OPEN  = 0.26
    MIN_CLOSE = 1
    MAX_CLOSE = 15

    passed = False
    if state == 'OPEN':
        if ear < EAR_CLOSE:
            state, frames = 'CLOSING', 1
    elif state == 'CLOSING':
        if ear < EAR_CLOSE:
            frames += 1
            if frames > MAX_CLOSE:          # held shut too long → reset
                state, frames = 'OPEN', 0
        else:                               # eye reopened
            if frames >= MIN_CLOSE:
                passed = True
            state, frames = 'OPEN', 0

    session['ch_blink_state']   = state
    session['ch_closed_frames'] = frames
    return passed


def check_turn_left(landmarks, frame_shape):
    """
    Head turned left (camera-left) when the nose tip is clearly
    to the LEFT of the face midpoint between both cheeks.
    """
    h, w, _ = frame_shape
    nose_x  = landmarks[NOSE_TIP].x * w
    left_x  = landmarks[LEFT_CHEEK].x * w
    right_x = landmarks[RIGHT_CHEEK].x * w
    mid_x   = (left_x + right_x) / 2.0
    # Nose must be shifted left of midpoint by at least 12 % of face width
    face_w  = abs(right_x - left_x)
    return (mid_x - nose_x) > (face_w * 0.12)


def check_turn_right(landmarks, frame_shape):
    h, w, _ = frame_shape
    nose_x  = landmarks[NOSE_TIP].x * w
    left_x  = landmarks[LEFT_CHEEK].x * w
    right_x = landmarks[RIGHT_CHEEK].x * w
    mid_x   = (left_x + right_x) / 2.0
    face_w  = abs(right_x - left_x)
    return (nose_x - mid_x) > (face_w * 0.12)


def check_smile(landmarks, frame_shape):
    """
    Smile ratio: mouth-width / mouth-height.
    A genuine smile widens the mouth significantly.
    """
    h, w, _ = frame_shape
    ml = (landmarks[MOUTH_LEFT].x  * w, landmarks[MOUTH_LEFT].y  * h)
    mr = (landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h)
    mt = (landmarks[MOUTH_TOP].x   * w, landmarks[MOUTH_TOP].y   * h)
    mb = (landmarks[MOUTH_BOTTOM].x* w, landmarks[MOUTH_BOTTOM].y* h)

    mouth_w = math.dist(ml, mr)
    mouth_h = math.dist(mt, mb) + 1e-6     # avoid zero-div
    ratio   = mouth_w / mouth_h
    return ratio > 3.8                      # tune between 3.5 – 4.5


def check_open_hand(hand_results, frame_shape):
    """
    Open palm: all 5 fingertips are ABOVE (smaller y) their base MCP joint.
    Also checks thumb is extended horizontally.
    """
    if not hand_results.hand_landmarks:
        return False

    lm   = hand_results.hand_landmarks[0]
    h, w, _ = frame_shape

    extended = 0
    for tip_id, mcp_id in zip(FINGERTIP_IDS[1:], FINGER_MCP_IDS[1:]):  # fingers 1-4
        if lm[tip_id].y < lm[mcp_id].y:   # tip above base in image coords
            extended += 1

    # Thumb: tip x farther from wrist than IP joint
    thumb_ok = abs(lm[4].x - lm[0].x) > abs(lm[3].x - lm[0].x)

    return extended >= 3 and thumb_ok      # at least 3 fingers + thumb extended


# ─────────────────────────────────────────────
#  SESSION DEFAULTS
# ─────────────────────────────────────────────

def init_challenge_session(session):
    defaults = {
        'liveness_passed':    False,
        'current_challenge':  None,
        'ch_blink_state':     'OPEN',
        'ch_closed_frames':   0,
        'recognition_count':  0,
        'last_recognized_id': None,
    }
    for k, v in defaults.items():
        if k not in session:
            session[k] = v


# ─────────────────────────────────────────────
#  API: assign a fresh random challenge
# ─────────────────────────────────────────────

@csrf_exempt
def assign_challenge(request):
    """Call this once when the camera opens to get the challenge for this session."""
    challenge = random.choice(CHALLENGES)

    # Reset any previous challenge state
    request.session['liveness_passed']   = False
    request.session['current_challenge'] = challenge['key']
    request.session['ch_blink_state']    = 'OPEN'
    request.session['ch_closed_frames']  = 0
    request.session['recognition_count'] = 0
    request.session['last_recognized_id']= None

    return JsonResponse({
        'challenge':    challenge['key'],
        'instruction':  challenge['instruction'],
        'emoji':        challenge['emoji'],
    })


# ─────────────────────────────────────────────
#  MAIN FRAME PROCESSOR
# ─────────────────────────────────────────────

DATASETS_DIR = os.path.join(settings.BASE_DIR, 'datasets')
TRAINER_DIR  = os.path.join(settings.BASE_DIR, 'trainer')
TRAINER_FILE = os.path.join(TRAINER_DIR, 'trainer.yml')


# Removed redundant get_current_recognizer logic


@csrf_exempt
def process_client_frame(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    try:
        data       = json.loads(request.body)
        image_data = data.get('image')
        if not image_data:
            return JsonResponse({'error': 'No image provided'}, status=400)

        # ── Decode frame ──────────────────────────────────────────────
        _, imgstr  = image_data.split(';base64,')
        frame      = cv2.imdecode(np.frombuffer(base64.b64decode(imgstr), np.uint8),
                                  cv2.IMREAD_COLOR)
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # ── Session bootstrap ─────────────────────────────────────────
        init_challenge_session(request.session)
        challenge       = request.session.get('current_challenge')
        liveness_passed = request.session.get('liveness_passed', False)

        # ── MediaPipe inference (Optimized: only run if needed) ───────
        face_results = None
        hand_results = None

        if not liveness_passed:
            if challenge in ['blink', 'turn_left', 'turn_right', 'smile']:
                face_results = face_landmarker.detect(mp_image)
            elif challenge == 'show_hand':
                hand_results = hand_landmarker.detect(mp_image)

        # ── Haar face boxes (for LBPH recognition) ────────────────────
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,  # Loosened from 1.3
                                              minNeighbors=4,         # Loosened from 5
                                              minSize=(30, 30))       # Loosened from 50x50

        if len(faces) > 0:
            print(f"[DEBUG] Detected {len(faces)} faces via Haar.")
        else:
            # Check if cascade is loaded
            if face_cascade.empty():
                print("[ERROR] Face cascade is EMPTY. Check XML path.")
            else:
                pass # No face in this frame

        challenge       = request.session.get('current_challenge')
        liveness_passed = request.session.get('liveness_passed', False)
        rec_count       = request.session['recognition_count']
        last_id         = request.session['last_recognized_id']

        # ── Challenge evaluation ──────────────────────────────────────
        challenge_result = {
            'passed':      liveness_passed,
            'challenge':   challenge,
        }

        if not liveness_passed and challenge:
            passed_this_frame = False

            if challenge in ['blink', 'turn_left', 'turn_right', 'smile'] and face_results and face_results.face_landmarks:
                lm = face_results.face_landmarks[0]

                if challenge == 'blink':
                    passed_this_frame = check_blink(lm, frame.shape, request.session)

                elif challenge == 'turn_left':
                    passed_this_frame = check_turn_left(lm, frame.shape)

                elif challenge == 'turn_right':
                    passed_this_frame = check_turn_right(lm, frame.shape)

                elif challenge == 'smile':
                    passed_this_frame = check_smile(lm, frame.shape)

            # Hand challenge doesn't need face landmarks
            elif challenge == 'show_hand' and hand_results:
                passed_this_frame = check_open_hand(hand_results, frame.shape)

            if passed_this_frame:
                print(f"[DEBUG] Challenge {challenge} PASSED.")
                liveness_passed = True
                request.session['liveness_passed'] = True
                challenge_result['passed'] = True

        # ── Face Recognition (LBPH) ───────────────────────────────────
        recognizer       = get_current_recognizer()
        drawing_commands = []
        current_id       = None

        for (x, y, fw, fh) in faces:
            label     = 'Scanning...'
            rgb_color = 'rgb(255,165,0)'

            # ── Preprocessing: improves LBPH accuracy significantly ───
            face_roi = gray[y:y+fh, x:x+fw]
            face_roi = cv2.resize(face_roi, (100, 100))             # normalize size
            face_roi = cv2.equalizeHist(face_roi)                   # fix lighting
            face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)        # reduce noise

            if recognizer is None:
                label, rgb_color = 'No Model', 'rgb(255,0,0)'
            else:
                try:
                    uid, confidence = recognizer.predict(face_roi)
                    print(f"[DEBUG] predict → uid={uid}, confidence={confidence:.1f}")

                    if confidence < 60:  # Loosened from 48 to 60
                        try:
                            profile    = UserProfile.objects.get(user_id=uid)
                            name       = profile.user.username
                            role       = profile.role.capitalize()
                            label      = f"{name} ({role}) [{confidence:.0f}]"
                            rgb_color  = ('rgb(0,0,255)'   if profile.role == 'teacher' else
                                          'rgb(255,165,0)' if profile.role == 'admin'   else
                                          'rgb(0,255,0)')
                            current_id = uid
                        except UserProfile.DoesNotExist:
                            label, rgb_color = 'Profile Missing', 'rgb(255,0,0)'
                    else:
                        label     = f'Unknown [{confidence:.0f}]'
                        rgb_color = 'rgb(255,0,0)'

                except cv2.error as e:
                    print(f"[ERROR] recognizer.predict failed: {e}")
                    label, rgb_color = 'Error', 'rgb(255,0,0)'

            drawing_commands.append({
                'type': 'rect',
                'x': int(x), 'y': int(y), 'w': int(fw), 'h': int(fh),
                'color': rgb_color, 'label': label,
            })

        # ── Recognition counter (only increments after liveness pass) ─
        if current_id and liveness_passed:
            if current_id == last_id:
                rec_count += 1
            else:
                last_id, rec_count = current_id, 1
        else:
            # Revert to 0 if liveness fails or person lost
            rec_count, last_id = 0, None

        # ── Persist session ───────────────────────────────────────────
        request.session['recognition_count']  = rec_count
        request.session['last_recognized_id'] = last_id

        return JsonResponse({
            'status':            'success',
            'draw':              drawing_commands,
            'liveness':          liveness_passed,
            'challenge':         challenge_result,
            'recognition_count': rec_count,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


# ─────────────────────────────────────────────
#  RESET  (call when camera closes or user retries)
# ─────────────────────────────────────────────

# Duplicate reset_recognition_session removed from here


@csrf_exempt
def stop_camera(request):
    reset_recognition_session(request)
    return JsonResponse({'status': 'Session cleared'})
# --- Authentication Views ---
def home(request):
    return render(request, 'homepage.html')

def login_page(request):
    # If already logged in, go straight to dashboard
    if request.user.is_authenticated:
        return redirect('dashboard')
        
    if request.method == 'POST':
        u_name = request.POST.get('username')
        p_word = request.POST.get('password')
        
        user = authenticate(request, username=u_name, password=p_word)
        
        if user is not None:
            # --- Prevent Concurrent Logins ---
            from django.contrib.sessions.models import Session
            from django.utils import timezone
            active_sessions = Session.objects.filter(expire_date__gte=timezone.now())
            
            force_login = request.POST.get('force_login') == 'true'
            session_conflict = False
            sessions_to_delete = []

            for session in active_sessions:
                data = session.get_decoded()
                if str(user.pk) == str(data.get('_auth_user_id')):
                    if force_login:
                        sessions_to_delete.append(session.session_key)
                    else:
                        session_conflict = True
            
            if session_conflict:
                messages.error(request, 'This account is currently logged in on another device.')
                return render(request, 'login.html', {'session_conflict': True, 'u_name': u_name})
                
            if force_login and sessions_to_delete:
                Session.objects.filter(session_key__in=sessions_to_delete).delete()
                messages.success(request, 'Previous sessions terminated.')
                    
            # Check if it's a teacher or student for 2FA
            is_teacher = hasattr(user, 'teacher_profile')
            is_student = hasattr(user, 'userprofile') and user.userprofile.role == 'student'
            is_admin = user.is_superuser or (hasattr(user, 'userprofile') and user.userprofile.role == 'admin')
            
            if is_teacher or is_student:
                # Check if face login is disabled by admin
                face_login_setting = AppSetting.objects.filter(name='require_face_login').first()
                if face_login_setting and not face_login_setting.is_enabled:
                    # Skip 2FA completely
                    login(request, user)
                    messages.success(request, f"Logged in as {u_name} (Face Login Disabled)")
                    return redirect('dashboard')
                
                # Otherwise, proceed with 2FA
                request.session['pending_2fa_user_id'] = user.id
                return redirect('face_scan_login')
            
            # Admins bypass 2FA
            login(request, user)
            messages.success(request, f"Logged in as {u_name}")
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password')
            
    return render(request, 'login.html')

def face_scan_login_view(request):
    if 'pending_2fa_user_id' not in request.session:
        return redirect('login')
    # We can reuse the teacher 2FA template but generalized
    return render(request, 'dashboard/teacher_face_scan.html', {'show_sidebar': False})

def verify_teacher_face(request):
    pending_id = request.session.get('pending_2fa_user_id')
    if not pending_id:
        return JsonResponse({'status': 'error', 'message': 'No pending login'})

    liveness_passed = request.session.get('liveness_passed', False)
    last_recognized_id = request.session.get('last_recognized_id')
    recognition_count = request.session.get('recognition_count', 0)
    
    if not liveness_passed:
        challenge = request.session.get('current_challenge', 'action')
        return JsonResponse({'status': 'pending', 'message': f'Please complete the {challenge} challenge'})
        
    if not (last_recognized_id and recognition_count >= 3):
        return JsonResponse({'status': 'pending'})
        
    if last_recognized_id == pending_id:
        try:
            user = User.objects.get(id=pending_id)
            login(request, user)
            reset_recognition_session(request)
            return JsonResponse({'status': 'success', 'url': '/dashboard/'})
        except User.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'User not found'})
            
    return JsonResponse({'status': 'pending'})


def backfill_absences():
    """
    Checks all dates from first recorded attendance to yesterday.
    If a student has no record for a given past date, backfills an 'Absent' record.
    """
    from django.utils import timezone
    from datetime import timedelta
    
    today = timezone.localdate()
    
    # Get the earliest attendance record date
    first_record = Attendance.objects.order_by('date').first()
    if not first_record:
        return # No attendance ever taken, nothing to backfill
        
    start_date = first_record.date
    if start_date >= today:
        return # Only backfill for PAST dates
        
    all_students = Student.objects.all()
    
    # Iterate through each past date from start_date to yesterday
    current_date = start_date
    while current_date < today:
        # Get IDs of students who DO have attendance on this day
        present_student_ids = Attendance.objects.filter(date=current_date).values_list('student_id', flat=True)
        
        # Any student not in that list needs an absence record
        missing_students = all_students.exclude(id__in=present_student_ids)
        
        # Bulk create absent records for efficiency
        absent_records = []
        for student in missing_students:
            absent_records.append(Attendance(
                student=student,
                date=current_date,
                time=timezone.make_aware(datetime.combine(current_date, timezone.localtime().time().replace(hour=23, minute=59, second=59))),
                status='Absent'
            ))
            
        if absent_records:
            Attendance.objects.bulk_create(absent_records)
            
        current_date += timedelta(days=1)


@login_required
def dashboard_redirect(request):
    """Central hub to separate users by role after login."""
    
    # Run the lazy auto-absence backfill exactly once per session/dashboard visit
    backfill_absences()
    
    try:
        profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        # Auto-create profile if missing (helps with old admin accounts)
        new_role = 'admin' if request.user.is_superuser else 'student'
        profile = UserProfile.objects.create(
            user=request.user, 
            role=new_role, 
            face_id=request.user.id
        )
    
    if profile.role == 'admin' or request.user.is_superuser:
        return redirect('admin_dashboard')
    elif profile.role == 'teacher':
        return redirect('teacher_dashboard')
    elif profile.role == 'student':
        return redirect('student_dashboard')
    
    return redirect('home')

@login_required
def admin_dashboard(request):
    # Only Admin/Superuser sees the admin control center
    if not (request.user.is_superuser or (hasattr(request.user, 'userprofile') and request.user.userprofile.role == 'admin')):
        return redirect('dashboard')
    
    student_count = Student.objects.count()
    return render(request, 'dashboard/admin_dash.html', {
        'show_sidebar': True,
        'student_count': student_count
    })

@login_required
def manage_settings(request):
    if not (request.user.is_superuser or (hasattr(request.user, 'userprofile') and request.user.userprofile.role == 'admin')):
        return redirect('dashboard')
    
    # Auto-create the built-in settings if they don't exist yet
    DEFAULTS = [
        ('require_face_login', True, 'When enabled, students and teachers must verify their face via camera after entering their password. Disable to allow password-only login.'),
        ('auto_mark_absent', True, 'When enabled, students who did not scan their face on any past date are automatically marked as Absent when any dashboard is visited.'),
        ('allow_manual_attendance', True, 'When enabled, teachers and admins can manually add or edit attendance records from the management panel.'),
        ('student_pdf_export', True, 'When enabled, students can export their own attendance history as a PDF report.'),
    ]
    for name, default_enabled, desc in DEFAULTS:
        AppSetting.objects.get_or_create(
            name=name,
            defaults={'is_enabled': default_enabled, 'description': desc}
        )
    
    if request.method == 'POST':
        setting_id = request.POST.get('setting_id')
        try:
            setting = AppSetting.objects.get(id=setting_id)
            setting.is_enabled = not setting.is_enabled
            setting.save()
            status_text = 'enabled' if setting.is_enabled else 'disabled'
            messages.success(request, f"'{setting.name}' has been {status_text}.")
        except AppSetting.DoesNotExist:
            messages.error(request, 'Setting not found.')
        return redirect('manage_settings')
    
    all_settings = AppSetting.objects.all().order_by('name')
    return render(request, 'management/settings.html', {
        'show_sidebar': True,
        'settings': all_settings
    })

@login_required
def manage_faculty(request):
    if not (request.user.is_superuser or request.user.userprofile.role == 'admin'):
        return redirect('dashboard')
    
    teachers = Teacher.objects.all()
    return render(request, 'management/manage_faculty.html', {
        'teachers': teachers,
        'show_sidebar': True
    })

@login_required
def teacher_dashboard(request):
    # Faculty dashboard
    if not (hasattr(request.user, 'userprofile') and request.user.userprofile.role == 'teacher'):
        return redirect('dashboard')
    return render(request, 'dashboard/teacher_dash.html', {'show_sidebar': True})

@login_required
def student_dashboard(request):
    # Student portal
    if not (hasattr(request.user, 'userprofile') and request.user.userprofile.role == 'student'):
        return redirect('dashboard')
        
    student = Student.objects.filter(user_profile__user=request.user).first()
    
    recent_attendance = []
    classes_missed = 0
    presence_percentage = 0
    start_date = request.GET.get('start_date', '')
    end_date = request.GET.get('end_date', '')
    
    if student:
        all_attendance_stats = Attendance.objects.filter(student=student)
        
        # Calculate stats unconditionally 
        total_days = all_attendance_stats.count()
        present_days = all_attendance_stats.filter(status='Present').count()
        classes_missed = all_attendance_stats.filter(status='Absent').count()
        
        if total_days > 0:
            presence_percentage = int((present_days / total_days) * 100)
            
        # Get history list (optionally filtered)
        query = all_attendance_stats
        if start_date and end_date:
            query = query.filter(date__range=[start_date, end_date])
            
        recent_attendance = query.order_by('-date', '-time')
        
        # Default to 15 records if no filter is applied
        if not start_date and not end_date:
            recent_attendance = recent_attendance[:15]
            
    return render(request, 'dashboard/student_dash.html', {
        'show_sidebar': True,
        'recent_attendance': recent_attendance,
        'classes_missed': classes_missed,
        'presence_percentage': presence_percentage,
        'start_date': start_date,
        'end_date': end_date
    })

@login_required
def student_profile(request):
    if not (hasattr(request.user, 'userprofile') and request.user.userprofile.role == 'student'):
        return redirect('dashboard')
        
    student = Student.objects.filter(user_profile__user=request.user).first()
    
    return render(request, 'dashboard/student_profile.html', {
        'show_sidebar': True,
        'student': student
    })

@login_required
def student_export_pdf(request):
    """Generates a PDF file of the logged-in student's attendance logs."""
    if not (hasattr(request.user, 'userprofile') and request.user.userprofile.role == 'student'):
        return HttpResponse("Unauthorized", status=403)
        
    student = Student.objects.filter(user_profile__user=request.user).first()
    if not student:
        return HttpResponse("Student record not found", status=404)
        
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    import io

    query = Attendance.objects.filter(student=student)
    
    start_date = request.GET.get('start_date', '')
    end_date = request.GET.get('end_date', '')
    if start_date and end_date:
        query = query.filter(date__range=[start_date, end_date])
        
    logs = query.order_by('-date', '-time')
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    
    elements.append(Paragraph(f"Attendance Report - {request.user.username}", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %b, %Y at %I:%M %p')}", styles['Normal']))
    elements.append(Spacer(1, 24))

    # Table Data
    data = [['Date', 'Time', 'Status']]
    for log in logs:
        data.append([
            str(log.date),
            log.time.strftime('%I:%M %p'),
            log.status
        ])

    t = Table(data, colWidths=[120, 120, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(t)
    doc.build(elements)
    
    pdf = buffer.getvalue()
    buffer.close()
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="my_attendance_report_{date.today()}.pdf"'
    response.write(pdf)
    
    return response

# --- Admin View: Register Teacher ---
@login_required
def register_teacher(request):
    # Only Admin can access this
    try:
        if request.user.userprofile.role != 'admin':
            return redirect('dashboard')
    except UserProfile.DoesNotExist:
        return redirect('dashboard')

    if request.method == 'POST':
        u_name = request.POST.get('username')
        p_word = request.POST.get('password')
        name = request.POST.get('name', u_name)
        dept = request.POST.get('department', 'General')
        
        if not User.objects.filter(username=u_name).exists():
            new_user = User.objects.create_user(username=u_name, password=p_word)
            profile = UserProfile.objects.create(
                user=new_user, 
                role='teacher', 
                face_id=new_user.id
            )
            # Create Teacher model entry
            Teacher.objects.create(
                user=new_user,
                name=name,
                department=dept
            )
            messages.success(request, f"Account created for {u_name}. Starting face capture...")
            request.session['enroll_user_id'] = new_user.id
            return redirect('enroll_face')
        else:
            messages.error(request, "Username already exists!")

    return render(request, 'dashboard/admin_register_teacher.html', {'show_sidebar': True})

@login_required
def register_student(request):
    # Only Admin can access this
    try:
        if request.user.userprofile.role != 'admin':
            return redirect('dashboard')
    except UserProfile.DoesNotExist:
        return redirect('dashboard')

    if request.method == 'POST':
        form = StudentRegistrationForm(request.POST)
        if form.is_valid():
            u_name = form.cleaned_data.get('username')
            p_word = form.cleaned_data.get('password')
            email = form.cleaned_data.get('email')
            roll_no = form.cleaned_data.get('roll_number')
            dept = form.cleaned_data.get('department')

            # Friendly duplicate checks before creating anything
            if Student.objects.filter(roll_number=roll_no).exists():
                messages.error(request, f"Roll number '{roll_no}' is already registered. Please use a different roll number.")
                return render(request, 'registration/register_student.html', {'form': form, 'show_sidebar': True})

            if User.objects.filter(username=u_name).exists():
                messages.error(request, f"Username '{u_name}' is already taken. Please choose a different username.")
                return render(request, 'registration/register_student.html', {'form': form, 'show_sidebar': True})

            # Create the User
            full_name = form.cleaned_data.get('name', '')
            print(f"[DEBUG] Creating student user: {u_name}, Name: {full_name}")
            
            new_user = User.objects.create_user(
                username=u_name, 
                password=p_word, 
                email=email,
                first_name=full_name
            )

            # Create UserProfile
            profile = UserProfile.objects.create(
                user=new_user,
                role='student',
                face_id=new_user.id
            )

            # Create Student
            Student.objects.create(
                user_profile=profile,
                roll_number=roll_no,
                department=dept
            )

            # Store ID in session for face enrollment
            request.session['enroll_user_id'] = new_user.id
            request.session.modified = True
            
            print(f"[DEBUG] Student {u_name} registered successfully. Redirecting to enroll_face.")
            messages.success(request, f"Details saved for {full_name}! Now let's capture the face.")
            return redirect('enroll_face')
        else:
            print(f"[DEBUG] Form invalid: {form.errors.as_json()}")
    else:
        form = StudentRegistrationForm()

    return render(request, 'registration/register_student.html', {
        'form': form,
        'show_sidebar': True
    })

@login_required
def enroll_face(request):
    if 'enroll_user_id' not in request.session:
        messages.error(request, "No student in enrollment queue.")
        return redirect('dashboard')
    return render(request, 'registration/enroll_face.html', {'show_sidebar': True})

def capture_progress_api(request):
    progress = request.session.get('capture_progress', 0)
    return JsonResponse({'progress': progress})

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

# --- Custom Management Views (No Admin Redirection) ---

@login_required
def view_all_students(request):
    if not (request.user.is_superuser or request.user.userprofile.role in ['admin', 'teacher']):
        return redirect('dashboard')
    
    students = Student.objects.all()
    user_role = request.user.userprofile.role if hasattr(request.user, 'userprofile') else 'admin'
    return render(request, 'management/all_students.html', {
        'students': students, 
        'show_sidebar': True,
        'user_role': user_role,
    })

@login_required
def todays_attendance(request):
    from datetime import date
    if not (request.user.is_superuser or request.user.userprofile.role in ['admin', 'teacher']):
        return redirect('dashboard')
    
    today = date.today()
    query_date = request.GET.get('date', today.strftime('%Y-%m-%d'))
    query_name = request.GET.get('student_name', '')
    query_month = request.GET.get('month', '')
    query_year = request.GET.get('year', '')
    
    logs = Attendance.objects.all()
    
    # Month/Year filter overrides the date filter
    if query_month and query_year:
        logs = logs.filter(date__year=query_year, date__month=query_month)
        query_date = ''  # Clear date when month/year is active
    elif query_year:
        logs = logs.filter(date__year=query_year)
        query_date = ''
    elif query_date:
        logs = logs.filter(date=query_date)
        
    if query_name:
        logs = logs.filter(student__user_profile__user__username__icontains=query_name)

    # Calculate unmarked students (only if a specific date is filtered, not month/year)
    unmarked_students = []
    if query_date and not (query_month and query_year):
        marked_student_ids = logs.values_list('student_id', flat=True)
        unmarked_students = Student.objects.exclude(id__in=marked_student_ids)

    # Build year range for dropdown (from 2020 to current year)
    years = list(range(today.year, 2019, -1))
    months = [
        {'num': 1, 'name': 'January'}, {'num': 2, 'name': 'February'},
        {'num': 3, 'name': 'March'}, {'num': 4, 'name': 'April'},
        {'num': 5, 'name': 'May'}, {'num': 6, 'name': 'June'},
        {'num': 7, 'name': 'July'}, {'num': 8, 'name': 'August'},
        {'num': 9, 'name': 'September'}, {'num': 10, 'name': 'October'},
        {'num': 11, 'name': 'November'}, {'num': 12, 'name': 'December'},
    ]

    return render(request, 'management/attendance_today.html', {
        'logs': logs, 
        'unmarked_students': unmarked_students,
        'show_sidebar': True,
        'current_date': query_date,
        'search_name': query_name,
        'current_month': int(query_month) if query_month else 0,
        'current_year': int(query_year) if query_year else 0,
        'years': years,
        'months': months,
    })

@login_required
def manual_attendance(request):
    if not (request.user.is_superuser or request.user.userprofile.role in ['admin', 'teacher']):
        return redirect('dashboard')
        
    if request.method == "POST":
        student_id = request.POST.get('student_id')
        status = request.POST.get('status', 'Present')
        try:
            student = Student.objects.get(id=student_id)
            Attendance.objects.create(student=student, status=status)
            messages.success(request, f"Manual attendance marked for {student.user_profile.user.username}")
        except Student.DoesNotExist:
            messages.error(request, "Student not found.")
            
        return redirect('todays_attendance')
        
    # Query all students for the manual entry form
    students = Student.objects.all()
    return render(request, 'management/manual_entry.html', {
        'students': students, 
        'show_sidebar': True
    })

@login_required
def mark_absent_quick(request):
    if not (request.user.is_superuser or request.user.userprofile.role in ['admin', 'teacher']):
        return redirect('dashboard')
        
    if request.method == "POST":
        student_id = request.POST.get('student_id')
        target_date = request.POST.get('target_date')
        
        try:
            from datetime import datetime
            student = Student.objects.get(id=student_id)
            # Create an Absent record for the specific date, at 11:59 PM to signify end-of-day missing
            Attendance.objects.create(
                student=student, 
                status='Absent',
                date=target_date,
                time=datetime.strptime("23:59:59", "%H:%M:%S").time()
            )
            messages.success(request, f"Marked {student.user_profile.user.username} as Absent for {target_date}.")
        except Student.DoesNotExist:
            messages.error(request, "Student not found.")
            
        # Redirect back to the same filtered view
        return redirect(f"/today-attendance/?date={target_date}")
        
    return redirect('todays_attendance')

@login_required
def mark_all_absent(request):
    if not (request.user.is_superuser or request.user.userprofile.role in ['admin', 'teacher']):
        return redirect('dashboard')
        
    if request.method == "POST":
        target_date = request.POST.get('target_date')
        
        if not target_date:
            return redirect('todays_attendance')
        
        try:
            from datetime import datetime
            
            # Find all marked students for this date
            marked_student_ids = Attendance.objects.filter(date=target_date).values_list('student_id', flat=True)
            unmarked_students = Student.objects.exclude(id__in=marked_student_ids)
            
            # Create absent records
            absent_records = []
            for student in unmarked_students:
                absent_records.append(
                    Attendance(
                        student=student, 
                        status='Absent',
                        date=target_date,
                        time=datetime.strptime("23:59:59", "%H:%M:%S").time()
                    )
                )
            
            if absent_records:
                Attendance.objects.bulk_create(absent_records)
                messages.success(request, f"Successfully marked {len(absent_records)} students as Absent for {target_date}.")
            else:
                messages.info(request, f"No unmarked students found for {target_date}.")
                
        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
            
        # Redirect back to the same filtered view
        return redirect(f"/today-attendance/?date={target_date}")
        
    return redirect('todays_attendance')

@login_required
def remove_student(request, student_id):
    if not (request.user.is_superuser or request.user.userprofile.role == 'admin'):
        return redirect('dashboard')
        
    student = get_object_or_404(Student, id=student_id)
    user_id = student.user_profile.user.id
    student.user_profile.user.delete() # This cascades to the profile and student record
    messages.success(request, f"Student {student.roll_number} and all associated data permanently removed.")
    return redirect('view_all_students')

@login_required
def remove_teacher(request, teacher_id):
    if not (request.user.is_superuser or request.user.userprofile.role == 'admin'):
        return redirect('dashboard')
        
    teacher = get_object_or_404(Teacher, id=teacher_id)
    teacher.user.delete() # Cascades to UserProfile and Teacher
    messages.success(request, f"Faculty member {teacher.name} permanently removed.")
    return redirect('manage_faculty')

def face_scan_page(request):
    return render(request, 'face_scan.html')

def logout_view(request):
    logout(request)
    return redirect('home')

# --- Face Capture and Training ---

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

        # Initialize progress & attempt counters in session
        if 'capture_progress' not in request.session:
            request.session['capture_progress'] = 0
            request.session['capture_attempts'] = 0

        progress = request.session['capture_progress']
        attempts = request.session.get('capture_attempts', 0)

        if progress >= 50:
            return JsonResponse({'status': 'complete', 'progress': 50})

        # Fail early if too many frames tried with no face detected
        MAX_ATTEMPTS = 120
        if attempts >= MAX_ATTEMPTS and progress == 0:
            # Clean up session
            request.session.pop('capture_progress', None)
            request.session.pop('capture_attempts', None)
            return JsonResponse({'status': 'error', 'message': 'No face detected. Please ensure your face is clearly visible and well-lit, then try again.'})

        format, imgstr = image_data.split(';base64,')
        img_bytes = base64.b64decode(imgstr)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if not os.path.exists(DATASETS_DIR):
            os.makedirs(DATASETS_DIR)

        for (x, y, w, h) in faces:
            progress += 1
            cv2.imwrite(os.path.join(DATASETS_DIR, f"User.{user_id}.{progress}.jpg"), gray[y:y+h, x:x+w])
            request.session['capture_progress'] = progress
            break  # only 1 face per frame

        attempts += 1
        request.session['capture_attempts'] = attempts

        return JsonResponse({'status': 'capturing', 'progress': progress})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def capture_face_samples(request, user_id):
    if not os.path.exists(DATASETS_DIR): os.makedirs(DATASETS_DIR)
    
    # We use a local camera object here to not interfere with the global stream
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    
    while True:
        ret, img = cam.read()
        if not ret: break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(os.path.join(DATASETS_DIR, f"User.{user_id}.{count}.jpg"), gray[y:y+h, x:x+w])
        
        if count >= 50: break
        time.sleep(0.05) 

    cam.release()
    cv2.destroyAllWindows() 
    
    if run_trainer_logic():
        try:
            profile = UserProfile.objects.get(user_id=user_id)
            profile.is_face_registered = True
            profile.save()
            messages.success(request, "Face registered and AI model trained!")
        except UserProfile.DoesNotExist:
            messages.error(request, "User profile not found after capture.")
    else:
        messages.error(request, "Training failed. No faces detected.")
        
    return redirect('dashboard')

def run_trainer_logic():
    path = DATASETS_DIR
    if not os.path.exists(path) or not os.listdir(path):
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=8, grid_x=8, grid_y=8  # more detailed grid
    )
    detector     = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    imagePaths   = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faceSamples, ids = [], []

    for imagePath in imagePaths:
        try:
            PIL_img   = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            uid       = int(os.path.split(imagePath)[-1].split(".")[1])
            faces     = detector.detectMultiScale(img_numpy, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = img_numpy[y:y+h, x:x+w]
                roi = cv2.resize(roi, (100, 100))      # ← must match predict size
                roi = cv2.equalizeHist(roi)            # ← same preprocessing
                roi = cv2.GaussianBlur(roi, (3, 3), 0)
                faceSamples.append(roi)
                ids.append(uid)
        except Exception as e:
            print(f"[TRAINER) Skipping {imagePath}: {e}")
            continue

    if ids:
        if not os.path.exists(TRAINER_DIR):
            os.makedirs(TRAINER_DIR)
        recognizer.train(faceSamples, np.array(ids))
        recognizer.write(TRAINER_FILE)

        # Bust the cache so next frame reloads fresh model
        global _recognizer_cache, _trainer_mtime
        _recognizer_cache = None
        _trainer_mtime    = None

        print(f"[TRAINER] Trained on {len(ids)} samples for {len(set(ids))} users.")
        return True

    print("[TRAINER] No face samples found.")
    return False

@csrf_exempt
def reset_recognition_session(request):
    keys = [
        'liveness_passed', 'current_challenge',
        'ch_blink_state', 'ch_closed_frames',
        'recognition_count', 'last_recognized_id',
        'blink_detected', 'ear_history',          # legacy keys
        'blink_state', 'closed_frame_count'       # legacy keys
    ]
    for k in keys:
        request.session.pop(k, None)
    return JsonResponse({'status': 'ok'})

def check_face_status(request):
    # ── NEW: require liveness to be passed first ──────────────────
    liveness_passed   = request.session.get('liveness_passed', False)
    last_recognized_id = request.session.get('last_recognized_id')
    recognition_count = request.session.get('recognition_count', 0)

    if not liveness_passed:
        challenge = request.session.get('current_challenge', 'the challenge')
        return JsonResponse({
            'status': 'pending',
            'message': f'Please complete the liveness check first ({challenge})'
        })

    if not (last_recognized_id and recognition_count >= 3):
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
        
        # --- CASE 1: LOGIN REDIRECTION (Not Authenticated) ---
        if not request.user.is_authenticated:
            # Fetch roll number for student
            if profile.role == 'student':
                try:
                    student = Student.objects.get(user_profile=profile)
                    roll_no = student.roll_number
                except Student.DoesNotExist:
                    roll_no = "No Roll No"

            login(request, user)
            
            # Use dynamic role-based redirect
            if profile.role == 'admin' or user.is_superuser:
                redirect_url = '/dashboard/admin/'
            elif profile.role == 'teacher':
                redirect_url = '/dashboard/teacher/'
            else:
                redirect_url = '/dashboard/student/'

            # Clear recognition to prevent multiple logins
            request.session['last_recognized_id'] = None
            request.session['recognition_count'] = 0
            
            return JsonResponse({
                'status': 'redirect',
                'name': name,
                'role': role,
                'roll_number': roll_no,
                'url': redirect_url
            })

        # --- CASE 2: ATTENDANCE MARKING (While Logged In) ---
        # Only students are marked for attendance
        if profile.role == 'student':
            try:
                student = Student.objects.get(user_profile=profile)
                roll_no = student.roll_number
                today = date.today()
                
                attendance_record = Attendance.objects.filter(student=student, date=today).first()
                
                if attendance_record:
                    return JsonResponse({
                        'status': 'already_marked',
                        'name': name,
                        'role': role,
                        'roll_number': roll_no,
                        'time': attendance_record.time.strftime('%I:%M %p'),
                        'date': attendance_record.date.strftime('%d %b, %Y')
                    })
                else:
                    # Mark as present for the first time today
                    new_record = Attendance.objects.create(
                        student=student,
                        date=today,
                        time=datetime.now().time(),
                        status="Present"
                    )
                    return JsonResponse({
                        'status': 'success_marked',
                        'name': name,
                        'role': role,
                        'roll_number': roll_no,
                        'time': new_record.time.strftime('%I:%M %p'),
                        'date': new_record.date.strftime('%d %b, %Y')
                    })
            except Student.DoesNotExist:
                pass

        # If it's another admin or teacher scanning themselves while logged in
        return JsonResponse({
            'status': 'identified_only',
            'name': name,
            'role': role,
            'roll_number': roll_no
        })

    except (User.DoesNotExist, UserProfile.DoesNotExist):
        pass
            
    return JsonResponse({'status': 'pending'})

# --- Reporting and Exports ---

@login_required
def export_reports_page(request):
    """Render the main export reports page with filters."""
    if not (request.user.is_superuser or request.user.userprofile.role in ['admin', 'teacher']):
        return redirect('dashboard')
    
    return render(request, 'management/export_reports.html', {
        'show_sidebar': True,
        'current_date': date.today().strftime('%d %b, %Y')
    })

def get_filtered_attendance(request):
    """Helper to filter attendance logs based on GET parameters."""
    logs = Attendance.objects.all().select_related('student__user_profile__user')
    
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    student_query = request.GET.get('student_query')

    if start_date and end_date:
        logs = logs.filter(date__range=[start_date, end_date])
    
    if student_query:
        # Search by student name (username) or roll number
        logs = logs.filter(
            Q(student__user_profile__user__username__icontains=student_query) |
            Q(student__roll_number__icontains=student_query)
        )
    
    return logs.order_by('-date', '-time')

@login_required
def export_attendance_csv(request):
    """Generates a CSV file of the attendance logs."""
    if not (request.user.is_superuser or request.user.userprofile.role in ['admin', 'teacher']):
        return HttpResponse("Unauthorized", status=403)

    logs = get_filtered_attendance(request)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="attendance_report_{date.today()}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Date', 'Time', 'Student Name', 'Roll Number', 'Department', 'Status'])
    
    for log in logs:
        writer.writerow([
            log.date,
            log.time.strftime('%I:%M %p'),
            log.student.user_profile.user.username,
            log.student.roll_number,
            log.student.department,
            log.status
        ])
    
    return response

@login_required
def export_attendance_pdf(request):
    """Generates a PDF file of the attendance logs using ReportLab."""
    if not (request.user.is_superuser or request.user.userprofile.role in ['admin', 'teacher']):
        return HttpResponse("Unauthorized", status=403)

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    import io

    logs = get_filtered_attendance(request)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    
    elements.append(Paragraph("SmartAttend - Attendance Report", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %b, %Y at %I:%M %p')}", styles['Normal']))
    elements.append(Spacer(1, 24))

    # Table Data
    data = [['Date', 'Time', 'Student Name', 'Roll No', 'Status']]
    for log in logs:
        data.append([
            str(log.date),
            log.time.strftime('%I:%M %p'),
            log.student.user_profile.user.username,
            log.student.roll_number,
            log.status
        ])

    t = Table(data, colWidths=[80, 80, 150, 100, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(t)
    doc.build(elements)
    
    pdf = buffer.getvalue()
    buffer.close()
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="attendance_report_{date.today()}.pdf"'
    response.write(pdf)
    
    return response

@login_required
def edit_student(request, student_id):
    if not (request.user.is_superuser or request.user.userprofile.role == 'admin'):
        return redirect('dashboard')

    from django.shortcuts import get_object_or_404
    student = get_object_or_404(Student, id=student_id)

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'rescan':
            student.user_profile.is_face_registered = False
            student.user_profile.save()
            request.session['enroll_user_id'] = student.user_profile.user.id
            messages.success(request, f"Ready to rescan face for {student.roll_number}")
            return redirect('enroll_face')

        elif action == 'save':
            u_name = request.POST.get('username')
            roll = request.POST.get('roll_number')
            dept = request.POST.get('department')

            if User.objects.filter(username=u_name).exclude(id=student.user_profile.user.id).exists():
                messages.error(request, "Username already in use.")
            elif Student.objects.filter(roll_number=roll).exclude(id=student.id).exists():
                messages.error(request, f"Roll number '{roll}' is already registered to another student.")
            else:
                student.user_profile.user.username = u_name
                if request.POST.get('password'):
                    student.user_profile.user.set_password(request.POST.get('password'))
                student.user_profile.user.save()
                student.roll_number = roll
                student.department = dept
                student.save()
                messages.success(request, "Student details updated successfully.")
                return redirect('view_all_students')

    # Collect captured dataset photos for this student
    user_id = student.user_profile.user.id
    photo_urls = []
    if os.path.exists(DATASETS_DIR):
        for fname in sorted(os.listdir(DATASETS_DIR)):
            if fname.startswith(f"User.{user_id}.") and fname.endswith('.jpg'):
                photo_urls.append(f"/datasets/{fname}")

    return render(request, 'management/edit_student.html', {
        'student': student,
        'show_sidebar': True,
        'photo_urls': photo_urls,
    })

@login_required
def edit_teacher(request, teacher_id):
    if not (request.user.is_superuser or request.user.userprofile.role == 'admin'):
        return redirect('dashboard')

    from django.shortcuts import get_object_or_404
    teacher = get_object_or_404(Teacher, id=teacher_id)

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'rescan':
            teacher.user.userprofile.is_face_registered = False
            teacher.user.userprofile.save()
            request.session['enroll_user_id'] = teacher.user.id
            messages.success(request, f"Ready to rescan face for {teacher.name}")
            return redirect('enroll_face')

        elif action == 'save':
            u_name = request.POST.get('username')
            name = request.POST.get('name')
            dept = request.POST.get('department')

            if User.objects.filter(username=u_name).exclude(id=teacher.user.id).exists():
                messages.error(request, "Username already in use.")
            else:
                teacher.user.username = u_name
                if request.POST.get('password'):
                    teacher.user.set_password(request.POST.get('password'))
                teacher.user.save()
                teacher.name = name
                teacher.department = dept
                teacher.save()
                messages.success(request, "Faculty details updated successfully.")
                return redirect('manage_faculty')

    # Collect captured dataset photos for this teacher
    user_id = teacher.user.id
    photo_urls = []
    if os.path.exists(DATASETS_DIR):
        for fname in sorted(os.listdir(DATASETS_DIR)):
            if fname.startswith(f"User.{user_id}.") and fname.endswith('.jpg'):
                photo_urls.append(f"/datasets/{fname}")

    return render(request, 'management/edit_teacher.html', {
        'teacher': teacher,
        'show_sidebar': True,
        'photo_urls': photo_urls,
    })

@login_required
def edit_attendance(request, record_id):
    if not (request.user.is_superuser or request.user.userprofile.role in ['admin', 'teacher']):
        return redirect('dashboard')
        
    from django.shortcuts import get_object_or_404
    record = get_object_or_404(Attendance, id=record_id)
    
    if request.method == 'POST':
        status = request.POST.get('status')
        if status in ['Present', 'Absent', 'Late']:
            record.status = status
            record.save()
            messages.success(request, f"Attendance record updated for {record.student.roll_number}.")
            return redirect('todays_attendance')

    return render(request, 'management/edit_attendance.html', {'record': record, 'show_sidebar': True})
