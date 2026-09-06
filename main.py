from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
from PIL import Image, ImageOps
import io

app = FastAPI()


def preprocess_image(image_bytes, max_size=1280, jpeg_quality=88):
    """Decode image, apply EXIF orientation, resize large photos, and compress safely."""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")

        width, height = pil_image.size
        largest = max(width, height)

        if largest > max_size:
            scale = max_size / float(largest)
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

        output = io.BytesIO()
        pil_image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
        processed = np.frombuffer(output.getvalue(), np.uint8)
        image = cv2.imdecode(processed, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image format")

        return image
    except Exception as e:
        raise ValueError(f"Unable to process image: {e}")


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
# DATABASE CONFIGURATION
# =====================================================

# Render PostgreSQL is the permanent database.
# JSON is kept only as a local fallback.
DATABASE_URL = os.getenv("DATABASE_URL")

STUDENT_DB = os.path.join(DATA_FOLDER, "students.json")
ATTENDANCE_DB = os.path.join(DATA_FOLDER, "attendance.json")

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:
    psycopg = None
    dict_row = None


def get_db_connection():
    if not DATABASE_URL or psycopg is None:
        return None
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def load_students_json():
    if not os.path.exists(STUDENT_DB):
        return {}
    try:
        with open(STUDENT_DB, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def save_students_json(students):
    with open(STUDENT_DB, "w", encoding="utf-8") as file:
        json.dump(students, file, indent=4)


def load_attendance_json():
    if not os.path.exists(ATTENDANCE_DB):
        return []
    try:
        with open(ATTENDANCE_DB, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return []


def save_attendance_json(attendance):
    with open(ATTENDANCE_DB, "w", encoding="utf-8") as file:
        json.dump(attendance, file, indent=4)


def init_database():
    """Create PostgreSQL tables and migrate old JSON data once."""
    conn = get_db_connection()
    if conn is None:
        print("⚠️ PostgreSQL not available; using JSON fallback")
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    roll_number TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    photo TEXT,
                    embedding JSONB NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    roll_number TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence DOUBLE PRECISION
                )
            """)
        conn.commit()
        print("✅ PostgreSQL connected and tables ready")
    finally:
        conn.close()

    migrate_json_to_postgres()


def migrate_json_to_postgres():
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM students")
            student_count = cur.fetchone()["count"]

            if student_count == 0 and os.path.exists(STUDENT_DB):
                try:
                    with open(STUDENT_DB, "r", encoding="utf-8") as file:
                        old_students = json.load(file)

                    migrated = 0
                    for roll, student in old_students.items():
                        embedding = student.get("embedding")
                        if not embedding:
                            continue

                        cur.execute("""
                            INSERT INTO students
                                (roll_number, name, photo, embedding)
                            VALUES (%s, %s, %s, %s::jsonb)
                            ON CONFLICT (roll_number) DO NOTHING
                        """, (
                            roll,
                            student.get("name", "Unknown"),
                            student.get("photo"),
                            json.dumps(embedding)
                        ))
                        migrated += 1

                    print(f"✅ Migrated {migrated} old student(s) to PostgreSQL")
                except Exception as e:
                    print("⚠️ Student JSON migration skipped:", e)

            cur.execute("SELECT COUNT(*) AS count FROM attendance")
            attendance_count = cur.fetchone()["count"]

            if attendance_count == 0 and os.path.exists(ATTENDANCE_DB):
                try:
                    with open(ATTENDANCE_DB, "r", encoding="utf-8") as file:
                        old_attendance = json.load(file)

                    migrated = 0
                    for record in old_attendance:
                        cur.execute("""
                            INSERT INTO attendance
                                (session_id, name, roll_number, date, time,
                                 status, confidence)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            record.get("session_id", ""),
                            record.get("name", "Unknown"),
                            record.get("roll_number", ""),
                            record.get("date", ""),
                            record.get("time", ""),
                            record.get("status", "Present"),
                            record.get("confidence")
                        ))
                        migrated += 1

                    print(f"✅ Migrated {migrated} attendance record(s) to PostgreSQL")
                except Exception as e:
                    print("⚠️ Attendance JSON migration skipped:", e)

        conn.commit()
    finally:
        conn.close()


def load_students():
    conn = get_db_connection()
    if conn is None:
        return load_students_json()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT roll_number, name, photo, embedding
                FROM students
                ORDER BY roll_number
            """)
            rows = cur.fetchall()

        return {
            row["roll_number"]: {
                "name": row["name"],
                "roll_number": row["roll_number"],
                "photo": row["photo"],
                "embedding": row["embedding"]
            }
            for row in rows
        }
    finally:
        conn.close()


def save_student(student):
    conn = get_db_connection()

    if conn is None:
        students = load_students_json()
        students[student["roll_number"]] = student
        save_students_json(students)
        print("⚠️ Student saved to JSON fallback:", student["roll_number"])
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO students
                    (roll_number, name, photo, embedding)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (roll_number)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    photo = EXCLUDED.photo,
                    embedding = EXCLUDED.embedding
            """, (
                student["roll_number"],
                student["name"],
                student.get("photo"),
                json.dumps(student["embedding"])
            ))
        conn.commit()
        print("✅ Student saved to PostgreSQL:", student["roll_number"])
    finally:
        conn.close()


def load_attendance():
    conn = get_db_connection()
    if conn is None:
        return load_attendance_json()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT session_id, name, roll_number, date, time,
                       status, confidence
                FROM attendance
                ORDER BY id ASC
            """)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def save_attendance_records(records):
    if not records:
        return

    conn = get_db_connection()

    if conn is None:
        attendance = load_attendance_json()
        attendance.extend(records)
        save_attendance_json(attendance)
        print("⚠️ Attendance saved to JSON fallback")
        return

    try:
        with conn.cursor() as cur:
            for record in records:
                cur.execute("""
                    INSERT INTO attendance
                        (session_id, name, roll_number, date, time,
                         status, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    record["session_id"],
                    record["name"],
                    record["roll_number"],
                    record["date"],
                    record["time"],
                    record["status"],
                    record["confidence"]
                ))
        conn.commit()
        print(f"✅ Saved {len(records)} attendance record(s) to PostgreSQL")
    finally:
        conn.close()

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

# Initialize permanent PostgreSQL storage.
init_database()


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

        image = preprocess_image(image_bytes)

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

        student_record = {
            "name": name,
            "roll_number": roll_number,
            "photo": photo_filename,
            "embedding": embedding
        }

        save_student(student_record)


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

        image = preprocess_image(image_bytes)

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

        save_attendance_records(newly_marked)


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

    data = load_students()

    students = []

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

    conn = get_db_connection()

    if conn is None:
        data = load_students_json()

        if roll_number not in data:
            return {
                "success": False,
                "message": "Student not found"
            }

        del data[roll_number]
        save_students_json(data)

    else:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM students WHERE roll_number = %s",
                    (roll_number,)
                )

                if cur.rowcount == 0:
                    return {
                        "success": False,
                        "message": "Student not found"
                    }

            conn.commit()
            print("✅ Student deleted from PostgreSQL:", roll_number)
        finally:
            conn.close()

    # Delete the local photo when it exists.
    # Attendance records remain preserved.
    safe_roll = "".join(
        c for c in roll_number
        if c.isalnum() or c in ("-", "_")
    )

    photo_path = os.path.join(
        STUDENTS_FOLDER,
        f"{safe_roll}.jpg"
    )

    if os.path.exists(photo_path):
        try:
            os.remove(photo_path)
        except Exception as e:
            print("⚠️ Could not remove photo:", e)

    return {
        "success": True,
        "message": "Student deleted successfully"
    }

