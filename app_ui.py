import streamlit as st
import tempfile
import cv2
import mediapipe as mp
import math
import numpy as np

# --- הגדרות עמוד ועיצוב ---
st.set_page_config(page_title="ניתוח זריקה מקצועי", layout="wide")

# עיצוב CSS ליישור לימין (RTL) ופונטים
st.markdown(
    """
    <style>
    .stApp {background-color:#FFFFFF; color:#000000;}
    h1, h2, h3, h4 {text-align:center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    
    /* יישור כל הטקסטים לימין */
    .stMarkdown, .stText, p, div {
        direction: rtl;
        text-align: right;
    }
    
    /* הסתרת כפתור מסך מלא של תמונות אם מפריע, או השארתו */
    button[title="View fullscreen"]{
        visibility: visible;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏀 AI Basketball Coach: ניתוח ביו-מכני כפול")
st.markdown("<div style='text-align:center; color:#666;'>ניתוח שלב הדריכה (Set Point) ושלב השחרור (Release)</div>", unsafe_allow_html=True)

# --- אתחול MediaPipe (מנגנון תיקון לענן) ---
# זה החלק שפותר את השגיאה AttributeError: module 'mediapipe' has no attribute 'solutions'
try:
    # ניסיון ראשון: יבוא סטנדרטי (עובד במחשב אישי)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    # ניסיון שני: יבוא ישיר לשרתי לינוקס/ענן
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- פונקציות עזר ---

def calculate_angle(a, b, c):
    """חישוב זווית בין 3 נקודות במישור דו-ממדי"""
    if a is None or b is None or c is None:
        return None
    
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    
    # מניעת חלוקה באפס
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return None
        
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_skeleton_on_image(frame, landmarks):
    """ציור השלד על גבי התמונה"""
    img_copy = frame.copy()
    mp_drawing.draw_landmarks(
        img_copy, 
        landmarks, 
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )
    return img_copy

# --- ממשק העלאת קובץ ---
uploaded_file = st.file_uploader("בחר סרטון (MP4/MOV)", type=['mp4', 'mov'], label_visibility="collapsed")

# --- הלוגיקה מרכזית ---
if uploaded_file is not None:
    # שמירת קובץ זמני
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
    
    # לולאת עיבוד הסרטון
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # איברים רלוונטיים (צד ימין כברירת מחדל)
            # 12=כתף, 14=מרפק, 16=שורש כף יד, 24=אגן, 26=ברך, 28=קרסול
            shoulder = lm[12]
            elbow = lm[14]
            wrist = lm[16]
            hip = lm[24]
            knee = lm[26]
            ankle = lm[28]
            
            # חישוב זוויות לפריים הנוכחי
            el_angle = calculate_angle(shoulder, elbow, wrist)
            kn_angle = calculate_angle(hip, knee, ankle)
            sh_angle = calculate_angle(hip, shoulder, elbow)
            
            frames_data.append({
                'frame': frame,
                'landmarks': results.pose_landmarks,
                'wrist_y': wrist.y,
                'knee_angle': kn_angle,
                'elbow_angle': el_angle,
                'shoulder_angle': sh_angle
            })
        else:
            frames_data.append(None)
            
    cap.release()
    progress_bar.empty()
    status_text.empty()

    # זיהוי פריימים מיוחדים
    valid_frames = [f for f in frames_data if f is not None]
    
    if not valid_frames:
        st.error("לא זוהה שחקן בסרטון.")
    else:
        # 1. זיהוי Release Point (יד הכי גבוהה)
        release_idx = min(range(len(valid_frames)), key=lambda i: valid_frames[i]['wrist_y'])
        release_data = valid_frames[release_idx]
        
        # 2. זיהוי Set Point (כיפוף ברכיים מקסימלי לפני השחרור)
        pre_release = valid_frames[:release_idx]
        if pre_release:
            set_idx = min(range
