import streamlit as st
import cv2
import numpy as np
import json
import os
import smtplib
import openpyxl
from email.mime.text import MIMEText
from PIL import Image
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import pandas as pd
from io import BytesIO
#هذي الكومنتات حق ملاك بس 
# ─────────────────────────────────────────
# Mediapipe
# ─────────────────────────────────────────
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode        = False,
        max_num_faces            = 1,
        min_detection_confidence = 0.5
    )
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]
except:
    face_mesh = None
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #1A0533; }
    h1, h2, h3, p, label, div { color: #EDE9F6 !important; }

    .stat-card {
        background: #2D1052;
        border-radius: 16px;
        padding: 18px 12px;
        text-align: center;
        border: 1px solid #6D28D9;
    }
    .stat-number { font-size: 40px; font-weight: bold; font-family: Georgia; }
    .stat-label  { font-size: 13px; color: #A78BFA !important; margin-top: 4px; }

    .stButton > button {
        background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 10px 28px !important;
        font-size: 15px !important;
        font-weight: bold !important;
        width: 100% !important;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #6D28D9, #5B21B6) !important; }

    .stTabs [data-baseweb="tab"] { font-size: 15px !important; font-weight: bold !important; color: #C4B5FD !important; }
    .stTabs [aria-selected="true"] { color: #A78BFA !important; border-bottom: 3px solid #A78BFA !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1A0533 !important; }

    .stTextInput input { background-color: #2D1052 !important; color: #EDE9F6 !important; border: 1px solid #6D28D9 !important; border-radius: 10px !important; }

    .stDownloadButton > button { background: linear-gradient(135deg, #059669, #047857) !important; color: white !important; border-radius: 12px !important; border: none !important; font-weight: bold !important; width: 100% !important; }

    hr { border-color: #4A1080 !important; }

    input[type="time"] { color: #000000 !important; }
    .stTimeInput input { color: #000000 !important; }
    [data-testid="stTimeInput"] input { color: #000000 !important; }

</style>

""", unsafe_allow_html=True)


# المسارات الباث 
# ─────────────────────────────────────────
BASE        = r'D:\Dev\datascience-course\final_project2'
EXCEL_PATH  = r'D:\Dev\datascience-course\final_project2\Students_Images\Students.xlsx'
REPORT_PATH = r'D:\Dev\datascience-course\final_project2\Students_Images\attendance_report.xlsx'

# من الفاينل بروجكت8 Model
# ─────────────────────────────────────────
model             = load_model(os.path.join(BASE, 'face_model.keras'))
feature_extractor = tf.keras.models.Model(
    inputs  = model.input,
    outputs = model.layers[-2].output
)

with open(os.path.join(BASE, 'student_features.json')) as f:
    raw = json.load(f)
student_features = {name: np.array(feat) for name, feat in raw.items()}


# الدوال الي باستخدمهم حق ابني المشروع
# ─────────────────────────────────────────
def load_students_db():
    if os.path.exists(EXCEL_PATH):
        return pd.read_excel(EXCEL_PATH)
    return pd.DataFrame(columns=['Name', 'Phone', 'Parent Email', 'Image Path'])

def load_img(path):
    img = Image.open(path).convert('RGB').resize((128, 128))
    return np.array(img).astype(np.float32) / 255.0

def refresh_features():
    global student_features
    with open(os.path.join(BASE, 'student_features.json')) as f:
        raw = json.load(f)
    student_features = {name: np.array(feat) for name, feat in raw.items()}

def identify_face(frame):
    img  = Image.fromarray(frame).convert('RGB').resize((128, 128))
    img  = np.array(img).astype(np.float32) / 255.0
    img  = np.expand_dims(img, axis=0)
    pred = feature_extractor.predict(img, verbose=0).flatten()
    best_name, best_score = None, 0
    for name, feat in student_features.items():
        score = cosine_similarity(pred.reshape(1,-1), feat.reshape(1,-1))[0][0]
        if score > best_score:
            best_score, best_name = score, name
    print(f'Best match: {best_name} | Score: {best_score:.4f}')
    return best_name if best_score >= 0.6 else None

def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def detect_drowsiness(frame):
    if face_mesh is None:
        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        faces        = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return 'No Face'
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        eye_region  = gray[y:y+int(h*0.6), x:x+w]
        eyes        = eye_cascade.detectMultiScale(eye_region, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))
        return 'Active' if len(eyes) >= 2 else 'Sleepy'
    h, w    = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return 'No Face'
    landmarks = results.multi_face_landmarks[0].landmark
    left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    ear       = (left_ear + right_ear) / 2.0
    return 'Sleepy' if ear < 0.25 else 'Active'

def send_email(parent_email, student_name, status):
    try:
        sender   = 'gastudent2026@gmail.com'
        password = 'uiesndkttdauzxgs'
        msg      = MIMEText(f'Dear Parent,\n\nYour child {student_name} was marked as {status} today {datetime.now().strftime("%Y-%m-%d")}.\n\nRegards,\nAttendance System')
        msg['Subject'] = f'Attendance Notification - {student_name} - {status}'
        msg['From']    = sender
        msg['To']      = parent_email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print(f'Email sent to {parent_email}')
        return True
    except Exception as e:
        print(f'Email error: {e}')
        return False

def generate_report(logged, drowsiness_log):
    wb       = openpyxl.Workbook()
    ws       = wb.active
    ws.title = 'Attendance Report'
    ws.append(['Name', 'Date', 'Arrival Time', 'Status', 'Drowsiness'])
    date_str = datetime.now().strftime('%Y-%m-%d')
    for name in student_features:
        info       = logged.get(name, {'status': 'Absent', 'arrival': '---'})
        drowsiness = drowsiness_log.get(name, '---')
        ws.append([name, date_str, info['arrival'], info['status'], drowsiness])
    wb.save(REPORT_PATH)
    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer


# Session State-حق ستريملت
# ─────────────────────────────────────────
if 'logged'         not in st.session_state:
    st.session_state.logged         = {}
if 'drowsiness_log' not in st.session_state:
    st.session_state.drowsiness_log = {}

# Header-حق استريملت 
# ─────────────────────────────────────────
col_img, col_txt = st.columns([1, 4])
with col_img:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
with col_txt:
    st.markdown(f"""
    <div style="padding-top:10px">
        <h2 style="color:#C4B5FD; font-family:Georgia; margin:0">Smart Attendance System</h2>
        <p style="color:#A78BFA; font-size:13px; margin:4px 0 0 0">Face Recognition &nbsp;·&nbsp; Drowsiness Detection &nbsp;·&nbsp; Auto Email &nbsp;·&nbsp; Excel Report &nbsp;|&nbsp; {datetime.now().strftime('%A, %d %B %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('---')

tab1, tab2 = st.tabs(['Take Attendance', 'Add New Student'])

# التاب 1 — تسجيل الحضور
# ─────────────────────────────────────────
with tab1:

    st.subheader('Session Settings')
    col1, col2 = st.columns(2)
    with col1:
        present_time = st.time_input('Present before', value=datetime.strptime('08:00', '%H:%M').time())
    with col2:
        late_time = st.time_input('Late after', value=datetime.strptime('08:15', '%H:%M').time())

    st.markdown('---')

    if st.button('Open Camera', key='camera_btn'):
        refresh_features()
        cap         = cv2.VideoCapture(0)
        box         = st.empty()
        label       = st.empty()
        state_box   = st.empty()
        stop_camera = st.checkbox('Stop Camera', key='stop_check')
        count       = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box.image(rgb, channels='RGB', use_container_width=True)

            if count % 30 == 0:
                name       = identify_face(rgb)
                drowsiness = detect_drowsiness(frame)

                if name:
                    st.session_state.drowsiness_log[name] = drowsiness
                    state_box.info(f'{name} — {drowsiness}')
                    if name not in st.session_state.logged:
                        now     = datetime.now().time()
                        status  = 'Present' if now <= present_time else 'Late'
                        arrival = datetime.now().strftime('%H:%M:%S')
                        st.session_state.logged[name] = {'status': status, 'arrival': arrival}
                        label.success(f'{name} | {status} | {arrival}')
                    else:
                        label.info(f'{name} already logged')
                else:
                    label.warning('No face recognized')

            if stop_camera or cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

       
        # تسجيل الغائبين وارسال الايميل للجميع
        # ─────────────────────────────────────────
        db = load_students_db()
        for name in student_features:
            if name not in st.session_state.logged:
                st.session_state.logged[name] = {'status': 'Absent', 'arrival': '---'}
            parent_row = db[db['Name'] == name]
            if not parent_row.empty:
                parent_email = str(parent_row.iloc[0].get('Parent Email', ''))
                if parent_email and parent_email != 'nan':
                    status = st.session_state.logged[name]['status']
                    send_email(parent_email, name, status)

        st.success('Session ended!')


    
    # الإحصائيات
    # ─────────────────────────────────────────
    if st.session_state.logged:
        present = sum(1 for v in st.session_state.logged.values() if v['status'] == 'Present')
        late    = sum(1 for v in st.session_state.logged.values() if v['status'] == 'Late')
        absent  = sum(1 for v in st.session_state.logged.values() if v['status'] == 'Absent')
        total   = len(st.session_state.logged)
        sleepy  = sum(1 for v in st.session_state.drowsiness_log.values() if v == 'Sleepy')

        st.subheader('Session Summary')
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#34D399">{present}</div><div class="stat-label">Present</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#FBBF24">{late}</div><div class="stat-label">Late</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#F87171">{absent}</div><div class="stat-label">Absent</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#A78BFA">{total}</div><div class="stat-label">Total</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#FB923C">{sleepy}</div><div class="stat-label">Sleepy</div></div>', unsafe_allow_html=True)

        
        # اهني جدول الحضور والغياب
        # ─────────────────────────────────────────
        st.subheader('Attendance Table')
        rows = []
        for n, v in st.session_state.logged.items():
            rows.append({
                'Name'      : n,
                'Status'    : v['status'],
                'Arrival'   : v['arrival'],
                'Drowsiness': st.session_state.drowsiness_log.get(n, '---')
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        
        #  the report + email
        # ─────────────────────────────────────────
        st.markdown('---')
        col_email, col_download = st.columns(2)

        with col_email:
            if st.button('Send Emails to Parents', key='email_btn'):
                db         = load_students_db()
                sent_count = 0
                for name, info in st.session_state.logged.items():
                    parent_row = db[db['Name'] == name]
                    if not parent_row.empty:
                        parent_email = str(parent_row.iloc[0].get('Parent Email', ''))
                        if parent_email and parent_email != 'nan':
                            result = send_email(parent_email, name, info['status'])
                            if result:
                                sent_count += 1
                st.success(f'Emails sent: {sent_count}')

        with col_download:
            report_file = generate_report(st.session_state.logged, st.session_state.drowsiness_log)
            st.download_button(
                label     = 'Download Attendance Report',
                data      = report_file,
                file_name = f'attendance_report_{datetime.now().strftime("%Y-%m-%d")}.xlsx',
                mime      = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key       = 'download_btn'
            )


# التاب 2 — إضافة طالب جديد
# ─────────────────────────────────────────
with tab2:
    st.subheader('Add New Student')
    name_input         = st.text_input('Student Name',  key='new_name')
    phone_input        = st.text_input('Phone Number',  key='new_phone')
    parent_email_input = st.text_input('Parent Email',  key='new_parent_email')

    photo_option = st.radio('Photo Option', ['Upload Photo', 'Take Photo'], key='photo_option', horizontal=True)

    if photo_option == 'Upload Photo':
        photo_input = st.file_uploader('Upload Photo', type=['jpg', 'jpeg', 'png'], key='new_photo')
        img_source  = photo_input
    else:
        photo_input = st.camera_input('Take a Photo', key='camera_photo')
        img_source  = photo_input

    if st.button('Add Student', key='add_btn'):
        if name_input and phone_input and parent_email_input and img_source:
            img_path = os.path.join(BASE, 'Students_Images', f'{name_input}.jpg')
            with open(img_path, 'wb') as f:
                f.write(img_source.read())

            img      = load_img(img_path)
            img      = np.expand_dims(img, axis=0)
            features = feature_extractor.predict(img, verbose=0).flatten()
            student_features[name_input] = features

            embeddings = {n: feat.tolist() for n, feat in student_features.items()}
            with open(os.path.join(BASE, 'student_features.json'), 'w') as f:
                json.dump(embeddings, f)

            db = load_students_db()
            new_row = pd.DataFrame([{
                'Name': name_input, 'Phone': phone_input,
                'Parent Email': parent_email_input, 'Image Path': img_path,
                'Date': '', 'Arrival_Time': '', 'Status': '', 'Drowsiness': ''
            }])
            db = pd.concat([db, new_row], ignore_index=True)
            db.to_excel(EXCEL_PATH, index=False)
            st.success(f'{name_input} added successfully!')
        else:
            st.warning('Please fill all fields and add a photo')