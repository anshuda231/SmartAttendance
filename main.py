from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid

app = FastAPI()

# =====================================================
# FOLDERS
# =====================================================

STUDENTS_FOLDER = "students"
MODELS_FOLDER = "models"
DATA_FOLDER = "data"

os.makedirs(STUDENTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)


# =====================================================
# MODEL PATHS
# =====================================================

YUNET_MODEL = os.path.join(
    MODELS_FOLDER,
    "face_detection_yunet_2023mar.onnx"
)

SFACE_MODEL = os.path.join(
    MODELS_FOLDER,
    "face_recognition_sface_2021dec.onnx"
)


# =====================================================
# DATABASE FILES
# =====================================================

STUDENT_DB = os.path.join(
    DATA_FOLDER,
    "students.json"
)

ATTENDANCE_DB = os.path.join(
    DATA_FOLDER,
    "attendance.json"
)


# =====================================================
# LOAD MODELS
# =====================================================

detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.70,
    0.3,
    5000
)

print("✅ YuNet face detector loaded")


recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)

print("✅ SFace face recognizer loaded")


# =====================================================
# STUDENT DATABASE FUNCTIONS
# =====================================================

def load_students():

    if not os.path.exists(STUDENT_DB):
        return {}

    try:

        with open(
            STUDENT_DB,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except:

        return {}


def save_students(students):

    with open(
        STUDENT_DB,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            students,
            file,
            indent=4
        )


# =====================================================
# ATTENDANCE DATABASE FUNCTIONS
# =====================================================

def load_attendance():

    if not os.path.exists(ATTENDANCE_DB):
        return []

    try:

        with open(
            ATTENDANCE_DB,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except:

        return []


def save_attendance(attendance):

    with open(
        ATTENDANCE_DB,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            attendance,
            file,
            indent=4
        )


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {
        "success": True,
        "message": "Smart Attendance Backend is Running"
    }


# =====================================================
# REGISTER STUDENT
# =====================================================

@app.post("/register-student")
async def register_student(
    name: str = Form(...),
    roll_number: str = Form(...),
    file: UploadFile = File(...)
):

    try:

        image_bytes = await file.read()

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return {
                "success": False,
                "message": "Invalid student photo"
            }


        # -------------------------------------------------
        # Detect face
        # -------------------------------------------------

        height, width = image.shape[:2]

        detector.setInputSize(
            (width, height)
        )

        _, faces = detector.detect(image)


        if faces is None or len(faces) == 0:

            return {
                "success": False,
                "message": "No face detected"
            }


        if len(faces) > 1:

            return {
                "success": False,
                "message":
                    "Please capture photo with only one face"
            }


        # -------------------------------------------------
        # Get face
        # -------------------------------------------------

        face = faces[0]


        # -------------------------------------------------
        # Align face
        # -------------------------------------------------

        aligned_face = recognizer.alignCrop(
            image,
            face
        )


        # -------------------------------------------------
        # Generate embedding
        # -------------------------------------------------

        feature = recognizer.feature(
            aligned_face
        )


        embedding = feature.flatten().tolist()


        # -------------------------------------------------
        # Clean roll number
        # -------------------------------------------------

        safe_roll = "".join(
            c for c in roll_number
            if c.isalnum()
        )

        if not safe_roll:

            return {
                "success": False,
                "message": "Invalid roll number"
            }


        # -------------------------------------------------
        # Save photo
        # -------------------------------------------------

        photo_filename = f"{safe_roll}.jpg"

        photo_path = os.path.join(
            STUDENTS_FOLDER,
            photo_filename
        )

        cv2.imwrite(
            photo_path,
            image
        )


        # -------------------------------------------------
        # Save student
        # -------------------------------------------------

        students = load_students()

        students[safe_roll] = {

            "name": name,

            "roll_number": roll_number,

            "photo": photo_filename,

            "embedding": embedding
        }

        save_students(students)


        print(
            f"✅ Student registered: "
            f"{name} | {roll_number}"
        )


        return {

            "success": True,

            "message":
                "Student registered successfully",

            "name": name,

            "roll_number": roll_number,

            "photo": photo_filename
        }


    except Exception as e:

        print(
            "❌ Registration error:",
            e
        )

        return {

            "success": False,

            "message": str(e)
        }


# =====================================================
# UPLOAD CLASSROOM PHOTO
# =====================================================

@app.post("/upload-photo")
async def upload_photo(
    file: UploadFile = File(...)
):

    try:

        # -------------------------------------------------
        # Read classroom image
        # -------------------------------------------------

        image_bytes = await file.read()

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return {
                "success": False,
                "message":
                    "Invalid classroom image"
            }


        # -------------------------------------------------
        # Detect faces
        # -------------------------------------------------

        height, width = image.shape[:2]

        detector.setInputSize(
            (width, height)
        )

        _, faces = detector.detect(image)


        if faces is None:

            faces = []


        print()
        print("📷 Classroom photo received")
        print(
            f"👤 Faces detected: {len(faces)}"
        )


        # -------------------------------------------------
        # Load students
        # -------------------------------------------------

        students = load_students()


        recognized_students = []


        # -------------------------------------------------
        # Recognize faces
        # -------------------------------------------------

        for face in faces:

            aligned_face = recognizer.alignCrop(
                image,
                face
            )

            feature = recognizer.feature(
                aligned_face
            )


            best_roll = None
            best_name = None
            best_score = -1


            # -------------------------------------------------
            # Compare with registered students
            # -------------------------------------------------

            for roll, student in students.items():

                stored_embedding = np.array(
                    student["embedding"],
                    dtype=np.float32
                ).reshape(1, -1)


                score = recognizer.match(
                    feature,
                    stored_embedding,
                    cv2.FaceRecognizerSF_FR_COSINE
                )


                if score > best_score:

                    best_score = score
                    best_roll = roll
                    best_name = student["name"]


            # -------------------------------------------------
            # Recognition threshold
            # -------------------------------------------------

            if best_score >= 0.30:

                recognized_students.append({

                    "name": best_name,

                    "roll_number": best_roll,

                    "confidence":
                        round(
                            float(best_score),
                            4
                        )
                })


        # =====================================================
        # MARK ATTENDANCE
        # =====================================================

        attendance = load_attendance()

        # Always save attendance time in India Standard Time (IST).
        now = datetime.now(ZoneInfo("Asia/Kolkata"))

        date = now.strftime(
            "%Y-%m-%d"
        )

        time = now.strftime(
            "%H:%M:%S"
        )

        # Every /upload-photo operation creates a NEW attendance session.
        # All students recognized from the same classroom photo share this
        # session_id. A different classroom photo gets a different ID.
        session_id = (
            now.strftime("%Y%m%d_%H%M%S_%f")
            + "_"
            + uuid.uuid4().hex[:8]
        )

        newly_marked = []

        # Prevent the same detected student from being added twice to
        # the SAME session if the detector happens to return duplicate faces.
        session_rolls = set()

        for student in recognized_students:

            roll_number = student["roll_number"]

            if roll_number in session_rolls:
                continue

            session_rolls.add(roll_number)

            # ---------------------------------------------
            # Create a NEW attendance record for this session
            # No same-day duplicate check is performed.
            # ---------------------------------------------

            record = {

                "session_id":
                    session_id,

                "name":
                    student["name"],

                "roll_number":
                    roll_number,

                "date":
                    date,

                "time":
                    time,

                "status":
                    "Present",

                "confidence":
                    student["confidence"]
            }

            attendance.append(record)
            newly_marked.append(record)


        # -------------------------------------------------
        # Save attendance
        # -------------------------------------------------

        save_attendance(attendance)


        print(
            f"✅ Attendance marked: "
            f"{len(newly_marked)} student(s)"
        )


        # =====================================================
        # RESPONSE
        # =====================================================

        return {

            "success": True,

            "message":
                "Attendance processed successfully",

            "faces_detected":
                len(faces),

            "recognized_students":
                recognized_students,

            "attendance_marked":
                newly_marked,

            "session_id":
                session_id
        }


    except Exception as e:

        print(
            "❌ Classroom processing error:",
            e
        )

        return {

            "success": False,

            "message": str(e)
        }


# =====================================================
# GET ATTENDANCE RECORDS
# =====================================================

@app.get("/attendance")
def get_attendance():

    attendance = load_attendance()

    return {

        "success": True,

        "attendance":
            attendance
    }
@app.get("/students")
def get_students():

    students = []

    if os.path.exists(STUDENT_DB):

        with open(STUDENT_DB, "r") as f:
            data = json.load(f)

        for roll_number, student in data.items():

            students.append({
                "name": student.get("name", "Unknown"),
                "roll_number": roll_number
            })

    return {
        "success": True,
        "students": students
    }
# ============================================================
# DELETE STUDENT
# ============================================================

@app.delete("/delete-student/{roll_number}")
def delete_student(roll_number: str):

    if not os.path.exists(STUDENT_DB):
        return {
            "success": False,
            "message": "No students registered"
        }

    with open(STUDENT_DB, "r") as f:
        data = json.load(f)

    if roll_number not in data:
        return {
            "success": False,
            "message": "Student not found"
        }

    # Remove student from face-recognition database
    del data[roll_number]

    with open(STUDENT_DB, "w") as f:
        json.dump(data, f, indent=4)

    # Remove saved student photo
    safe_roll = "".join(
        c for c in roll_number
        if c.isalnum() or c in ("-", "_")
    )

    photo_path = os.path.join(
        STUDENTS_FOLDER,
        f"{safe_roll}.jpg"
    )

    if os.path.exists(photo_path):
        os.remove(photo_path)

    return {
        "success": True,
        "message": "Student deleted successfully"
    }
