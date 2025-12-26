import streamlit as st
import tempfile
import cv2
import mediapipe as mp
import math
import numpy as np

# --- הגדרות עמוד ועיצוב ---
st.set_page_config(page_title="ניתוח זריקה מקצועי", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background-color:#FFFFFF; color:#000000;}
    h1, h2, h3, h4 {text-align:center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    .stMarkdown, .stText, p, div {direction: rtl; text-align: right;}
    button[title="View fullscreen"]{visibility: visible;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏀 AI Basketball Coach: ניתוח ביו-מכני כפול")
st.markdown("<div style='text-align:center; color:#666;'>ניתוח שלב הדריכה (Set Point) ושלב השחרור (Release)</div>", unsafe_allow_html=True)

# --- אתחול MediaPipe (מנגנון בטיחות לענן) ---
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- פונקציות עזר ---
def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return None
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_skeleton_on_image(frame, landmarks):
    img_copy = frame.copy()
    mp_drawing.draw_landmarks(
        img_copy, landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )
    return img_copy

# --- ממשק העלאה ---
uploaded_file = st.file_uploader("בחר סרטון (MP4/MOV)", type=['mp4', 'mov'], label_visibility="collapsed")

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.flush()
    
    cap = cv2.VideoCapture(tfile.name)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    frames_data = [] 
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("מעבד סרטון...")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if total_frames > 0: progress_bar.progress(min(frame_count / total_frames, 1.0))
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            frames_data.append({
                'frame': frame,
                'landmarks': results.pose_landmarks,
                'wrist_y': lm[16].y,
                'knee_angle': calculate_angle(lm[24], lm[26], lm[28]),
                'elbow_angle': calculate_angle(lm[12], lm[14], lm[16]),
                'shoulder_angle': calculate_angle(lm[24], lm[12], lm[14])
            })
        else:
            frames_data.append(None)
            
    cap.release()
    progress_bar.empty()
    status_text.empty()

    valid_frames = [f for f in frames_data if f
