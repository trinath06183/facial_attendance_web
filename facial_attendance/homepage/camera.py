import cv2
import threading
import time
import os

# Eye cascade is bundled with OpenCV — no extra packages needed
_EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'

# Sensitivity: how many consecutive frames eyes must be ABSENT to count as "closed"
CLOSED_FRAMES_MIN = 2


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        self.read_lock = threading.Lock()
        self.is_running = True
        self.last_recognized_id = None
        self.recognition_count = 0
        self.capture_progress = 0

        # Blink / Liveness state
        self.blink_detected = False
        self.blink_count = 0
        self.eye_closed_frames = 0   # consecutive frames where eyes not detected
        self._eyes_were_closed = False

        # Face detector (Haar cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Eye cascade for blink detection (built into OpenCV)
        self.eye_cascade = cv2.CascadeClassifier(_EYE_CASCADE_PATH)

        # LBPH recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists('trainer/trainer.yml'):
            try:
                self.recognizer.read('trainer/trainer.yml')
            except Exception as e:
                print(f"Error loading trainer: {e}")

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.is_running:
            (grabbed, frame) = self.video.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            time.sleep(0.01)

    def stop(self):
        self.is_running = False
        time.sleep(0.1)
        if self.video.isOpened():
            self.video.release()

    def reset_liveness(self):
        """Call after a successful recognition to reset for next session."""
        self.blink_detected = False
        self.blink_count = 0
        self.eye_closed_frames = 0
        self._eyes_were_closed = False
        self.last_recognized_id = None
        self.recognition_count = 0

    def _detect_blink(self, gray, face_rect):
        """
        Check for a blink in the eye region using the eye cascade.
        A blink = eyes absent for CLOSED_FRAMES_MIN frames, then reappear.
        """
        x, y, w, h = face_rect
        # Only look in the upper half of the face for eyes
        roi = gray[y: y + h // 2, x: x + w]
        eyes = self.eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        eyes_visible = len(eyes) > 0

        if not eyes_visible:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= CLOSED_FRAMES_MIN:
                # Eyes were closed for enough frames and are now open → blink!
                self.blink_count += 1
                self.blink_detected = True
            self.eye_closed_frames = 0

        return eyes_visible, eyes

    def get_frame(self):
        if not self.is_running:
            return None
        with self.read_lock:
            if not self.grabbed:
                return None
            image = self.frame.copy()

        image = cv2.flip(image, 1)
        image = cv2.resize(image, (640, 480))
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ── 1. FACE DETECTION ───────────────────────────────────────────────
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        current_recognized_id = None
        eyes_visible = False

        for (x, y, fw, fh) in faces:
            # ── 2. BLINK DETECTION via Eye Cascade ─────────────────────────
            eyes_visible, detected_eyes = self._detect_blink(gray, (x, y, fw, fh))

            # Eye markers (blink debug) removed

            # ── 3. FACE RECOGNITION via LBPH ───────────────────────────────
            try:
                user_id, confidence = self.recognizer.predict(gray[y:y+fh, x:x+fw])
                if confidence < 75:
                    from .models import UserProfile
                    profile = UserProfile.objects.get(user_id=user_id)
                    role = profile.role.capitalize()
                    name = profile.user.username
                    label = f"{name} ({role})"

                    if profile.role == 'teacher':
                        color = (255, 0, 0)
                    elif profile.role == 'admin':
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)

                    current_recognized_id = user_id
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
            except Exception:
                label = "Scanning..."
                color = (255, 165, 0)

            # Draw the face bounding box (rectangle as before)
            cv2.rectangle(image, (x, y), (x + fw, y + fh), color, 1)
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            cv2.rectangle(image, (x, y - text_h - 15), (x + max(fw, text_w + 10), y), color, cv2.FILLED)
            cv2.putText(image, label, (x + 5, y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # ── 4. UPDATE RECOGNITION COUNTER ──────────────────────────────────
        if current_recognized_id and self.blink_detected:
            if current_recognized_id == self.last_recognized_id:
                self.recognition_count += 1
            else:
                self.last_recognized_id = current_recognized_id
                self.recognition_count = 1
        elif not self.blink_detected:
            self.recognition_count = 0
            self.last_recognized_id = None

        # ── 5. LIVENESS HUD ────────────────────────────────────────────────
        if self.blink_detected:
            cv2.rectangle(image, (0, 440), (640, 480), (0, 80, 0), cv2.FILLED)
            cv2.putText(image, "  LIVENESS VERIFIED  -  Blink Detected!",
                        (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
        else:
            cv2.rectangle(image, (0, 440), (640, 480), (0, 50, 100), cv2.FILLED)
            cv2.putText(image, "  Please blink once to verify liveness...",
                        (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 200, 255), 2)

        # Debug bar: blink count and eye state
        eye_str = "Eyes: OPEN" if eyes_visible else f"Eyes: CLOSED ({self.eye_closed_frames}f)"
        cv2.putText(image, f"Blinks:{self.blink_count}  {eye_str}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def get_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists('trainer/trainer.yml'):
        try:
            recognizer.read('trainer/trainer.yml')
        except Exception as e:
            print(f"Error loading trainer: {e}")
    return recognizer
