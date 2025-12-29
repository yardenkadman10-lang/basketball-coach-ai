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
    
    /* --- עיצוב תיבות ה-FLOW --- */
    .flow-box-good {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #c3e6cb;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .flow-box-bad {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #f5c6cb;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏀 AI Basketball Coach: ניתוח ביו-מכני מתקדם")
st.markdown("<div style='text-align:center; color:#666;'>ניתוח דריכה (Dip), שחרור (Release) ושטף הזריקה (Flow)</div>", unsafe_allow_html=True)

# --- אתחול MediaPipe (בטוח לענן) ---
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- פונקציות חישוב ---
def calculate_angle(a, b, c):
    if a is None or b is None or c is None: return None
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return None
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_vertical_angle(a, b):
    if a is None or b is None: return None
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y) 
    angle_rad = math.atan2(dx, dy)
    return np.degrees(angle_rad)

def get_distance(a, b):
    if a is None or b is None: return 1000.0
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

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
    status_text.text("מנתח את הזריקה (Dip, Release, Flow)...")
    
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
            
            knee_ang = calculate_angle(lm[24], lm[26], lm[28]) 
            elbow_ang = calculate_angle(lm[12], lm[14], lm[16]) 
            wrist_ang = calculate_angle(lm[14], lm[16], lm[20]) 
            torso_ang = calculate_vertical_angle(lm[12], lm[24]) 
            shoulder_ang = calculate_angle(lm[24], lm[12], lm[14]) 
            
            # מרחק בין הידיים (15=שמאל, 16=ימין)
            hands_dist = get_distance(lm[15], lm[16])

            frames_data.append({
                'frame': frame,
                'landmarks': results.pose_landmarks,
                'wrist_y': lm[16].y, 
                'wrist_x': lm[16].x,
                'wrist_left_y': lm[15].y,
                'wrist_left_x': lm[15].x,
                'hands_dist': hands_dist,
                'knee_angle': knee_ang,
                'elbow_angle': elbow_ang,
                'wrist_angle': wrist_ang,
                'torso_angle': torso_ang,
                'shoulder_angle': shoulder_ang
            })
        else:
            frames_data.append(None)
            
    cap.release()
    progress_bar.empty()
    status_text.empty()

    valid_frames = [f for f in frames_data if f is not None]
    
    if not valid_frames:
        st.error("לא זוהה שחקן בסרטון.")
    else:
        # --- 1. זיהוי Release Point ---
        release_idx_in_valid = min(range(len(valid_frames)), key=lambda i: valid_frames[i]['wrist_y'])
        release_data = valid_frames[release_idx_in_valid]
        
        # --- 2. זיהוי Dip (Set Point) ---
        pre_release_frames = valid_frames[:release_idx_in_valid]
        set_data = None
        set_idx_in_valid = 0

        if pre_release_frames:
            # מציאת הנקודה הכי נמוכה של הכדור
            max_y_val = max(f['wrist_y'] for f in pre_release_frames)
            threshold = max_y_val * 0.95
            
            # סינון פריימים נמוכים
            lowest_indices = [
                i for i, f in enumerate(pre_release_frames) 
                if f['wrist_y'] >= threshold
            ]
            
            if lowest_indices:
                # בחירת הפריים עם האחיזה הכי טובה מבין הנמוכים
                best_local_idx = min(lowest_indices, key=lambda i: pre_release_frames[i]['hands_dist'])
                set_data = pre_release_frames[best_local_idx]
                set_idx_in_valid = best_local_idx
            else:
                # גיבוי
                best_local_idx = pre_release_frames.index(max(pre_release_frames, key=lambda f: f['wrist_y']))
                set_data = pre_release_frames[best_local_idx]
                set_idx_in_valid = best_local_idx
        else:
            set_data = valid_frames[0]
            set_idx_in_valid = 0

        # יצירת תמונות
        img_set = draw_skeleton_on_image(set_data['frame'], set_data['landmarks'])
        img_release = draw_skeleton_on_image(release_data['frame'], release_data['landmarks'])

        # --- תצוגה ויזואלית ---
        st.markdown("---")
        col1, col2 = st.columns(2, gap="large")

        # Set Point Column
        with col1:
            st.markdown("### 1️⃣ שלב ה-Dip (הכדור בנקודה נמוכה)")
            st.image(cv2.cvtColor(img_set, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown("#### 📊 ניתוח ביו-מכני:")
            
            ka = set_data['knee_angle']
            if 110 <= ka <= 120:
                st.markdown(f"✅ <b>ברכיים ({int(ka)}°):</b> טווח אידיאלי (110°-120°).", unsafe_allow_html=True)
            elif ka < 110:
                st.markdown(f"⚠️ <b>ברכיים ({int(ka)}°):</b> עמוק מדי.", unsafe_allow_html=True)
            else:
                st.markdown(f"❌ <b>ברכיים ({int(ka)}°):</b> זווית קהה מדי.", unsafe_allow_html=True)
            st.write("") 

            ta = set_data['torso_angle']
            if 0 <= ta <= 20: 
                st.markdown(f"✅ <b>גוף ({int(ta)}°):</b> יציב (0°-20°).", unsafe_allow_html=True)
            else:
                st.markdown(f"⚠️ <b>גוף ({int(ta)}°):</b> נטייה מוגזמת.", unsafe_allow_html=True)
            st.write("")

            ea = set_data['elbow_angle']
            if 75 <= ea <= 85:
                st.markdown(f"✅ <b>מרפק ({int(ea)}°):</b> L-Shape מדויק (75°-85°).", unsafe_allow_html=True)
            else:
                st.markdown(f"❌ <b>מרפק ({int(ea)}°):</b> חריגה (75°-85°).", unsafe_allow_html=True)
            st.write("")

            wa = set_data['wrist_angle']
            if 135 <= wa <= 165: 
                st.markdown(f"✅ <b>שורש כף יד ({int(wa)}°):</b> דריכה טובה.", unsafe_allow_html=True)
            else:
                st.markdown(f"⚠️ <b>שורש כף יד ({int(wa)}°):</b> לא בטווח ה-Snap.", unsafe_allow_html=True)


        # Release Point Column
        with col2:
            st.markdown("### 2️⃣ שלב השחרור (Release)")
            st.image(cv2.cvtColor(img_release, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown("#### 📊 ניתוח ביו-מכני:")

            ta_rel = release_data['torso_angle']
            if 0 <= ta_rel <= 20:
                st.markdown(f"✅ <b>גוף בשחרור ({int(ta_rel)}°):</b> יציבות מעולה (0°-20°).", unsafe_allow_html=True)
            else:
                st.markdown(f"❌ <b>גוף בשחרור ({int(ta_rel)}°):</b> חוסר יציבות.", unsafe_allow_html=True)
            st.write("")

            ea_rel = release_data['elbow_angle']
            if 160 <= ea_rel <= 180: 
                st.markdown(f"✅ <b>יישור מרפק ({int(ea_rel)}°):</b> סיומת מלאה.", unsafe_allow_html=True)
            else:
                st.markdown(f"❌ <b>יישור מרפק ({int(ea_rel)}°):</b> יד קצרה.", unsafe_allow_html=True)
            st.write("")

            ka_rel = release_data['knee_angle']
            if 160 <= ka_rel <= 180:
                st.markdown(f"✅ <b>יישור רגליים ({int(ka_rel)}°):</b> מעבר כוח מלא.", unsafe_allow_html=True)
            else:
                st.markdown(f"❌ <b>יישור רגליים ({int(ka_rel)}°):</b> הרגליים לא התיישרו.", unsafe_allow_html=True)
            st.write("")

            sa_rel = release_data['shoulder_angle']
            if sa_rel < 130: 
                 st.markdown(f"✅ <b>מרפק פנימי:</b> תקין (Elbow In).", unsafe_allow_html=True)
            else:
                 st.markdown(f"⚠️ <b>מרפק פנימי:</b> המרפק בורח החוצה.", unsafe_allow_html=True)

        # =========================================================
        # --- חישוב ה-FLOW והצגת התוצאה (כאן התוספת) ---
        # =========================================================
        
        # לוקחים את כל הפריימים מהדריכה (Set) ועד השחרור (Release)
        flow_frames = valid_frames[set_idx_in_valid : release_idx_in_valid + 1]
        
        is_flow_good = False
        flow_msg = "לא זוהו מספיק פריימים לחישוב שטף"

        if len(flow_frames) > 2:
            # פונקציה לחישוב תזוזה ממוצעת של שתי הידיים
            def calc_speed(f1, f2):
                d_r = np.sqrt((f1['wrist_x'] - f2['wrist_x'])**2 + (f1['wrist_y'] - f2['wrist_y'])**2)
                d_l = np.sqrt((f1['wrist_left_x'] - f2['wrist_left_x'])**2 + (f1['wrist_left_y'] - f2['wrist_left_y'])**2)
                return (d_r + d_l) / 2

            # מהירות הבסיס: הצעד הראשון של העלייה (בין ה-Dip לפריים שאחריו)
            base_speed = calc_speed(flow_frames[0], flow_frames[1])
            if base_speed < 0.001: base_speed = 0.001 # מניעת חלוקה באפס

            hitch_found = False
            
            # בודקים את כל שאר הפריימים בעלייה
            for i in range(1, len(flow_frames) - 1):
                curr_speed = calc_speed(flow_frames[i], flow_frames[i+1])
                
                # אם המהירות נופלת מתחת ל-10% ממהירות ההתחלה, זה נחשב "תקיעה" (Hitch)
                if curr_speed < base_speed * 0.15:
                    hitch_found = True
                    break
            
            if hitch_found:
                is_flow_good = False
                flow_msg = "ה-FLOW של הזריקה לא באותו קצב, נסה לשמור על ONE MOTION ליצירת זריקה יציבה"
            else:
                is_flow_good = True
                flow_msg = "ה-FLOW של הזריקה תקין, כל הכבוד"
        
        # --- הצגת התיבה המעוצבת ---
        st.markdown("---")
        st.markdown("### 🌊 ניתוח שטף הזריקה (Shot Flow)")
        
        if is_flow_good:
            st.markdown(f'<div class="flow-box-good">{flow_msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="flow-box-bad">{flow_msg}</div>', unsafe_allow_html=True)
