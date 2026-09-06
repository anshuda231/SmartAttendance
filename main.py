from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid

import psycopg
from psycopg.rows import dict_row

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
# DATABASE
# =====================================================

DATABASE_URL = os.getenv("DATABASE_URL")

STUDENT_DB = os.path.join(DATA_FOLDER, "students.json")
ATTENDANCE_DB = os.path.join(DATA_FOLDER, "attendance.json")


def db_enabled():
    return bool(DATABASE_URL)


def get_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_database():
    """Create PostgreSQL tables and import old JSON data once."""
    if not db_enabled():
        print("⚠️ DATABASE_URL not found. Using local JSON storage.")
        return

    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                roll_number TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                photo TEXT,
                embedding JSONB NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT,
                name TEXT NOT NULL,
                roll_number TEXT NOT NULL,
                date DATE NOT NULL,
                time TIME NOT NULL,
                status TEXT NOT NULL,
                confidence DOUBLE PRECISION
            )
        """)

        # Import existing JSON students only if PostgreSQL is empty.
        student_count = conn.execute(
            "SELECT COUNT(*) AS count FROM students"
        ).fetchone()["count"]

        if student_count == 0 and os.path.exists(STUDENT_DB):
            try:
                with open(STUDENT_DB, "r", encoding="utf-8") as f:
                    old_students = json.load(f)

                for roll, student in old_students.items():
                    conn.execute(
                        """
                        INSERT INTO students
                            (roll_number, name, photo, embedding)
                        VALUES
                            (%s, %s, %s, %s::jsonb)
                        ON CONFLICT (roll_number) DO NOTHING
                        """,
                        (
                            roll,
                            student.get("name", "Unknown"),
                            student.get("photo"),
                            json.dumps(student.get("embedding", []))
                        )
                    )

                print(f"✅ Imported {len(old_students)} student(s) from JSON")
            except Exception as e:
                print("⚠️ Student JSON migration skipped:", e)

        # Import old JSON attendance only if PostgreSQL is empty.
        attendance_count = conn.execute(
            "SELECT COUNT(*) AS count FROM attendance"
        ).fetchone()["count"]

        if attendance_count == 0 and os.path.exists(ATTENDANCE_DB):
            try:
                with open(ATTENDANCE_DB, "r", encoding="utf-8") as f:
                    old_attendance = json.load(f)

                for record in old_attendance:
                    record_date = record.get("date")
                    record_time = record.get("time")

                    if not record_date or not record_time:
                        continue

                    conn.execute(
                        """
                        INSERT INTO attendance
                            (session_id, name, roll_number, date, time,
                             status, confidence)
                        VALUES
                            (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            record.get("session_id"),
                            record.get("name", "Unknown"),
                            record.get("roll_number", ""),
                            record_date,
                            record_time,
                            record.get("status", "Present"),
                            record.get("confidence")
                        )
                    )

                print(
                    f"✅ Imported {len(old_attendance)} attendance record(s) "
                    "from JSON"
                )
            except Exception as e:
                print("⚠️ Attendance JSON migration skipped:", e)

        conn.commit()
        print("✅ PostgreSQL database ready")


# =====================================================
# STUDENT DATABASE FUNCTIONS
# =====================================================

def load_students():
    if db_enabled():
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT roll_number, name, photo, embedding
                FROM students
                ORDER BY roll_number
                """
            ).fetchall()

        result = {}
        for row in rows:
            embedding = row["embedding"]

            if isinstance(embedding, str):
                embedding = json.loads(embedding)

            result[row["roll_number"]] = {
                "name": row["name"],
                "roll_number": row["roll_number"],
                "photo": row["photo"],
                "embedding": embedding
            }

        return result

    if not os.path.exists(STUDENT_DB):
        return {}

    try:
        with open(STUDENT_DB, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def save_student(student):
    if db_enabled():
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO students
                    (roll_number, name, photo, embedding)
                VALUES
                    (%s, %s, %s, %s::jsonb)
                ON CONFLICT (roll_number)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    photo = EXCLUDED.photo,
                    embedding = EXCLUDED.embedding
                """,
                (
                    student["roll_number"],
                    student["name"],
                    student.get("photo"),
                    json.dumps(student["embedding"])
                )
            )
            conn.commit()

        return

    students = load_students()
    students[student["roll_number"]] = student

    with open(STUDENT_DB, "w", encoding="utf-8") as file:
        json.dump(students, file, indent=4)


# =====================================================
# ATTENDANCE DATABASE FUNCTIONS
# =====================================================

def load_attendance():
    if db_enabled():
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    session_id,
                    name,
                    roll_number,
                    TO_CHAR(date, 'YYYY-MM-DD') AS date,
                    TO_CHAR(time, 'HH24:MI:SS') AS time,
                    status,
                    confidence
                FROM attendance
                ORDER BY date DESC, time DESC, id DESC
                """
            ).fetchall()

        result = []

        for row in rows:
            result.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "name": row["name"],
                "roll_number": row["roll_number"],
                "date": row["date"],
                "time": row["time"],
                "status": row["status"],
                "confidence": row["confidence"]
            })

        return result

    if not os.path.exists(ATTENDANCE_DB):
        return []

    try:
        with open(ATTENDANCE_DB, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return []


def save_attendance_record(record):
    if db_enabled():
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO attendance
                    (session_id, name, roll_number, date, time,
                     status, confidence)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    record.get("session_id"),
                    record["name"],
                    record["roll_number"],
                    record["date"],
                    record["time"],
                    record["status"],
                    record.get("confidence")
                )
            )
            conn.commit()

        return

    attendance = load_attendance()
    attendance.append(record)

    with open(ATTENDANCE_DB, "w", encoding="utf-8") as file:
        json.dump(attendance, file, indent=4)


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


@app.on_event("startup")
def startup_event():
    try:
        init_database()
    except Exception as e:
        print("❌ Database startup error:", e)


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():
    return {
        "success": True,
        "message": "Smart Attendance Backend is Running",
        "database": "PostgreSQL" if db_enabled() else "Local JSON"
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

        height, width = image.shape[:2]

        detector.setInputSize((width, height))
        _, faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            return {
                "success": False,
                "message": "No face detected"
            }

        if len(faces) > 1:
            return {
                "success": False,
                "message": "Please capture photo with only one face"
            }

        face = faces[0]

        aligned_face = recognizer.alignCrop(
            image,
            face
        )

        feature = recognizer.feature(
            aligned_face
        )

        embedding = feature.flatten().tolist()

        safe_roll = "".join(
            c for c in roll_number
            if c.isalnum() or c in ("-", "_")
        )

        if not safe_roll:
            return {
                "success": False,
                "message": "Invalid roll number"
            }

        photo_filename = f"{safe_roll}.jpg"

        photo_path = os.path.join(
            STUDENTS_FOLDER,
            photo_filename
        )

        cv2.imwrite(photo_path, image)

        student = {
            "name": name,
            "roll_number": safe_roll,
            "photo": photo_filename,
            "embedding": embedding
        }

        save_student(student)

        print(
            f"✅ Student registered: {name} | {safe_roll}"
        )

        return {
            "success": True,
            "message": "Student registered successfully",
            "name": name,
            "roll_number": safe_roll,
            "photo": photo_filename
        }

    except Exception as e:
        print("❌ Registration error:", e)

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
                "message": "Invalid classroom image"
            }

        height, width = image.shape[:2]

        detector.setInputSize((width, height))
        _, faces = detector.detect(image)

        if faces is None:
            faces = []

        print()
        print("📷 Classroom photo received")
        print(f"👤 Faces detected: {len(faces)}")

        students = load_students()
        recognized_students = []

        for face in faces:
            aligned_face = recognizer.alignCrop(
                image,
                face
            )

            feature = recognizer.feature(
                aligned_face
            )

            # =================================================
            # STRICT FACE RECOGNITION
            # =================================================
            # SFace cosine similarity:
            # Higher score = more similar.
            #
            # 0.30 was too permissive and could create false
            # positives. We now require:
            #   1) a strong absolute similarity score
            #   2) a clear margin over the second-best student
            #
            # A face that does not satisfy BOTH conditions is
            # treated as unknown and is NOT marked Present.
            RECOGNITION_THRESHOLD = 0.40
            MIN_MATCH_MARGIN = 0.05

            matches = []

            for roll, student in students.items():
                stored_embedding = np.array(
                    student["embedding"],
                    dtype=np.float32
                ).reshape(1, -1)

                score = float(recognizer.match(
                    feature,
                    stored_embedding,
                    cv2.FaceRecognizerSF_FR_COSINE
                ))

                matches.append({
                    "roll_number": roll,
                    "name": student["name"],
                    "score": score
                })

            if matches:
                matches.sort(key=lambda x: x["score"], reverse=True)

                best_match = matches[0]
                best_score = best_match["score"]

                second_score = (
                    matches[1]["score"]
                    if len(matches) > 1
                    else -1.0
                )

                margin = best_score - second_score

                print(
                    f"🔎 Best match: {best_match['name']} "
                    f"({best_score:.4f}), "
                    f"second: {second_score:.4f}, "
                    f"margin: {margin:.4f}"
                )

                # IMPORTANT:
                # Do NOT mark attendance unless the face is both
                # similar enough AND clearly better than the next
                # candidate.
                if (
                    best_score >= RECOGNITION_THRESHOLD
                    and (
                        len(matches) == 1
                        or margin >= MIN_MATCH_MARGIN
                    )
                ):
                    recognized_students.append({
                        "name": best_match["name"],
                        "roll_number": best_match["roll_number"],
                        "confidence": round(best_score, 4)
                    })
                else:
                    print(
                        f"⚠️ Unknown face rejected: "
                        f"best_score={best_score:.4f}, "
                        f"margin={margin:.4f}"
                    )

        # =================================================
        # MARK ATTENDANCE
        # =================================================

        now = datetime.now(ZoneInfo("Asia/Kolkata"))

        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        session_id = (
            now.strftime("%Y%m%d_%H%M%S_%f")
            + "_"
            + uuid.uuid4().hex[:8]
        )

        newly_marked = []
        session_rolls = set()

        for student in recognized_students:
            roll_number = student["roll_number"]

            if roll_number in session_rolls:
                continue

            session_rolls.add(roll_number)

            record = {
                "session_id": session_id,
                "name": student["name"],
                "roll_number": roll_number,
                "date": date,
                "time": time,
                "status": "Present",
                "confidence": student["confidence"]
            }

            save_attendance_record(record)
            newly_marked.append(record)

        print(
            f"✅ Attendance marked: {len(newly_marked)} student(s)"
        )

        return {
            "success": True,
            "message": "Attendance processed successfully",
            "faces_detected": len(faces),
            "recognized_students": recognized_students,
            "attendance_marked": newly_marked,
            "session_id": session_id
        }

    except Exception as e:
        print("❌ Classroom processing error:", e)

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
        "attendance": attendance
    }


# =====================================================
# GET STUDENTS
# =====================================================

@app.get("/students")
def get_students():
    students_data = load_students()

    students = []

    for roll_number, student in students_data.items():
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
    try:
        if db_enabled():
            with get_db() as conn:
                row = conn.execute(
                    """
                    SELECT roll_number
                    FROM students
                    WHERE roll_number = %s
                    """,
                    (roll_number,)
                ).fetchone()

                if not row:
                    return {
                        "success": False,
                        "message": "Student not found"
                    }

                conn.execute(
                    "DELETE FROM students WHERE roll_number = %s",
                    (roll_number,)
                )

                conn.commit()

        else:
            if not os.path.exists(STUDENT_DB):
                return {
                    "success": False,
                    "message": "No students registered"
                }

            with open(STUDENT_DB, "r", encoding="utf-8") as f:
                data = json.load(f)

            if roll_number not in data:
                return {
                    "success": False,
                    "message": "Student not found"
                }

            del data[roll_number]

            with open(STUDENT_DB, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

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

    except Exception as e:
        print("❌ Delete student error:", e)

        return {
            "success": False,
            "message": str(e)
        }
