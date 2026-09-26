from __future__ import annotations

import io
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, desc, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from werkzeug.security import check_password_hash, generate_password_hash


# ============================================================================
# APP CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "final_edupredict_model.pkl"
FEATURES_PATH = BASE_DIR / "artifacts" / "feature_columns.pkl"
FINAL_DATA_PATH = BASE_DIR / "Data" / "final_selected_features.csv"
RAW_DATA_PATH = BASE_DIR / "Data" / "merged_dataset.csv"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "edupredict.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

MODEL_EXPECTED_FEATURES = [
    "Academic_Engagement",
    "Assignment_Attendance_Index",
    "Study_Assignment_Index",
    "Study_Attendance_Index",
    "AssignmentCompletion",
    "Attendance_Normalized",
    "Age",
    "Attendance",
    "StudyHours",
    "OnlineCourses",
    "StudyHours_Normalized",
    "LearningStyle",
    "Motivation_Study_Index",
    "StressLevel",
    "Online_Learning_Activity",
    "Participation_Rate",
    "Digital_Learning_Index",
    "Motivation",
]

RAW_INPUT_COLUMNS = [
    "StudyHours",
    "Attendance",
    "Resources",
    "Extracurricular",
    "Motivation",
    "Internet",
    "Gender",
    "Age",
    "LearningStyle",
    "OnlineCourses",
    "Discussions",
    "AssignmentCompletion",
    "EduTech",
    "StressLevel",
]

MODEL_CATEGORICAL_FEATURES = ["LearningStyle", "StressLevel", "Motivation"]
MODEL_NUMERICAL_FEATURES = [c for c in MODEL_EXPECTED_FEATURES if c not in MODEL_CATEGORICAL_FEATURES]

MAX_STUDY_HOURS = 44
MAX_ONLINE_COURSES = 20
MAX_RESOURCES = 2
MAX_LEARNING_STYLE = 3
MAX_MOTIVATION = 2
MAX_STRESS_LEVEL = 2

CLASS_LABELS = {
    0: "Highest historical performance group",
    1: "Upper-middle historical performance group",
    2: "Lower-middle historical performance group",
    3: "Lowest historical performance group",
}

CLASS_EXPLANATION = {
    0: "Class 0 represents the highest historical performance group in this dataset.",
    1: "Class 1 represents the upper-middle historical performance group in this dataset.",
    2: "Class 2 represents the lower-middle historical performance group in this dataset.",
    3: "Class 3 represents the lowest historical performance group in this dataset.",
}


# Teacher-facing labels are based on observable academic behavior and historical
# performance patterns. The supplied dataset does not contain a validated IQ measure.
SUPPORT_PROFILES = {
    0: {
        "title": "Strong, consistent academic pattern",
        "summary": "Historically this group shows the strongest overall performance pattern in the dataset.",
        "teacher_plan": "Keep normal attendance expectations. Use regular check-ins, enrichment, harder practice, and independent work where appropriate.",
    },
    1: {
        "title": "Generally strong, with some growth areas",
        "summary": "Historically this group sits above the middle of the dataset, with some areas that may need reinforcement.",
        "teacher_plan": "Keep normal attendance expectations. Use standard teaching plus targeted feedback on the weakest observable habits or features.",
    },
    2: {
        "title": "Developing pattern; targeted support helps",
        "summary": "Historically this group falls below the upper groups and may benefit from more structured academic support.",
        "teacher_plan": "Keep normal attendance expectations. Add structured practice, smaller milestones, frequent feedback, and focused support on weak observable features.",
    },
    3: {
        "title": "Needs closer academic support",
        "summary": "Historically this group has the lowest overall performance pattern in the dataset.",
        "teacher_plan": "Keep normal attendance expectations. Increase check-ins, provide specific intervention, break work into smaller steps, and coordinate follow-up with the student.",
    },
}

OBSERVABLE_FEATURES = [
    "StudyHours",
    "Attendance",
    "AssignmentCompletion",
    "Academic_Engagement",
    "Participation_Rate",
    "OnlineCourses",
    "Digital_Learning_Index",
    "Motivation_Study_Index",
]

OBSERVABLE_FEATURE_LABELS = {
    "StudyHours": "Study effort",
    "Attendance": "Attendance",
    "AssignmentCompletion": "Assignment completion",
    "Academic_Engagement": "Academic engagement",
    "Participation_Rate": "Participation",
    "OnlineCourses": "Online learning activity",
    "Digital_Learning_Index": "Digital learning access/use",
    "Motivation_Study_Index": "Motivation + study effort",
}

MODEL_METRICS = {
    "Test Accuracy": 0.9118,
    "Macro Precision": 0.9126,
    "Macro Recall": 0.9112,
    "Macro F1": 0.9118,
    "Cross-validation Macro F1": 0.8639,
}

CLASS_METRICS = pd.DataFrame(
    [
        [0, 0.9033, 0.9206, 0.9119, 680],
        [1, 0.9077, 0.9015, 0.9046, 589],
        [2, 0.9052, 0.9193, 0.9122, 644],
        [3, 0.9342, 0.9036, 0.9186, 581],
    ],
    columns=["FinalGrade", "Precision", "Recall", "F1", "Support"],
)

# Supplied project analysis output from the model notebook's permutation analysis.
PERMUTATION_IMPORTANCE = pd.DataFrame(
    [
        ["Study_Assignment_Index", 0.058449, 0.004818],
        ["Academic_Engagement", 0.046441, 0.003982],
        ["Assignment_Attendance_Index", 0.038229, 0.003482],
        ["Age", 0.026072, 0.002342],
        ["Study_Attendance_Index", 0.019153, 0.002921],
        ["OnlineCourses", 0.010611, 0.002136],
        ["Attendance", 0.009641, 0.001985],
        ["Attendance_Normalized", 0.009317, 0.001479],
        ["AssignmentCompletion", 0.008386, 0.001720],
        ["LearningStyle", 0.007516, 0.002236],
        ["StressLevel", 0.006768, 0.001286],
        ["StudyHours_Normalized", 0.005400, 0.001156],
        ["StudyHours", 0.004584, 0.000759],
        ["Digital_Learning_Index", 0.003255, 0.001255],
        ["Online_Learning_Activity", 0.002946, 0.001420],
        ["Motivation_Study_Index", 0.002135, 0.001112],
        ["Participation_Rate", 0.002113, 0.000899],
        ["Motivation", 0.000606, 0.000721],
    ],
    columns=["Feature", "Importance_Mean", "Importance_STD"],
)

FEATURE_HELP = {
    "Academic_Engagement": "Composite indicator based on assignment completion, attendance, normalized study hours, and discussion participation.",
    "Assignment_Attendance_Index": "Combines assignment completion and attendance.",
    "Study_Assignment_Index": "Combines normalized study hours with assignment completion.",
    "Study_Attendance_Index": "Combines normalized study hours with normalized attendance.",
    "AssignmentCompletion": "Percentage of assignments completed.",
    "Attendance_Normalized": "Attendance represented on a 0–1 scale.",
    "Age": "Student age.",
    "Attendance": "Student attendance percentage.",
    "StudyHours": "Study-hours value represented in the training data.",
    "OnlineCourses": "Number of online-course activities represented in the dataset.",
    "StudyHours_Normalized": "Study hours represented on a 0–1 scale.",
    "LearningStyle": "Encoded categorical feature with categories 0–3. The source data does not define semantic style names.",
    "Motivation_Study_Index": "Combined representation of motivation and study effort.",
    "StressLevel": "Encoded categorical feature with categories 0–2.",
    "Online_Learning_Activity": "Combined online-course activity with internet and educational-technology indicators.",
    "Participation_Rate": "Normalized participation based on extracurricular and discussion participation.",
    "Digital_Learning_Index": "Combined representation of online courses, internet availability, educational technology, and resources.",
    "Motivation": "Encoded categorical feature with categories 0–2.",
}

FEATURE_DESCRIPTIONS = {
    "Academic_Engagement": "Composite indicator based on assignment completion, attendance, normalized study hours, and discussion participation.",
    "Assignment_Attendance_Index": "Combines assignment completion and attendance.",
    "Study_Assignment_Index": "Combines normalized study hours with assignment completion.",
    "Study_Attendance_Index": "Combines normalized study hours and attendance.",
    "AssignmentCompletion": "Percentage of assignments completed.",
    "Attendance_Normalized": "Attendance represented on a 0–1 scale.",
    "Age": "Student age.",
    "Attendance": "Student attendance percentage.",
    "StudyHours": "Study-hours value represented in the training data.",
    "OnlineCourses": "Number of online-course activities represented in the dataset.",
    "StudyHours_Normalized": "Study hours represented on a 0–1 scale.",
    "LearningStyle": "Categorical variable encoded as 0, 1, 2, or 3; semantic names are not defined by the source data.",
    "Motivation_Study_Index": "Combined representation of motivation and study effort.",
    "StressLevel": "Categorical variable encoded as 0, 1, or 2.",
    "Online_Learning_Activity": "Combined online-course activity with internet and educational-technology indicators.",
    "Participation_Rate": "Normalized participation based on extracurricular and discussion participation.",
    "Digital_Learning_Index": "Combined representation of online courses, internet availability, educational technology, and resources.",
    "Motivation": "Categorical variable encoded as 0, 1, or 2.",
}

# ============================================================================
# STYLE
# ============================================================================

st.markdown(
    """
    <style>
        :root {
            --bg: #07111f;
            --panel: #0d1b2e;
            --panel-2: #10233b;
            --border: rgba(255,255,255,.09);
            --text: #eef5ff;
            --muted: #96a8bd;
            --accent: #63a4ff;
            --accent-2: #8b7cff;
            --success: #47d7a0;
            --warning: #ffc857;
        }
        .stApp {
            background:
                radial-gradient(circle at 15% 5%, rgba(99,164,255,.13), transparent 28%),
                radial-gradient(circle at 88% 12%, rgba(139,124,255,.12), transparent 30%),
                linear-gradient(180deg, #06101d 0%, #081523 55%, #091827 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07111f 0%, #0a1627 100%);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * { color: #dce8f7; }
        .hero {
            padding: 30px 32px;
            border: 1px solid var(--border);
            border-radius: 26px;
            background:
                linear-gradient(135deg, rgba(99,164,255,.14), rgba(139,124,255,.08)),
                rgba(13,27,46,.84);
            box-shadow: 0 18px 55px rgba(0,0,0,.24);
            margin-bottom: 22px;
        }
        .hero-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .1em;
            text-transform: uppercase;
            background: rgba(99,164,255,.12);
            border: 1px solid rgba(99,164,255,.28);
            color: #9ec9ff;
            margin-bottom: 13px;
        }
        .hero h1 {
            color: white;
            font-size: 43px;
            line-height: 1.04;
            letter-spacing: -.03em;
            margin: 0;
        }
        .hero p {
            color: #a9bad0;
            font-size: 16px;
            max-width: 960px;
            margin: 12px 0 0 0;
            line-height: 1.65;
        }
        .kpi {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 19px 18px;
            background: rgba(13,27,46,.78);
            box-shadow: 0 12px 35px rgba(0,0,0,.18);
            min-height: 120px;
        }
        .kpi-label {
            color: #91a6bd;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: .09em;
            font-weight: 800;
        }
        .kpi-value {
            color: white;
            font-size: 29px;
            font-weight: 850;
            margin-top: 7px;
        }
        .kpi-sub { color: #7188a3; font-size: 12px; margin-top: 5px; }
        .section-title {
            color: white;
            font-size: 21px;
            font-weight: 850;
            margin: 19px 0 8px 0;
        }
        .section-subtitle {
            color: #91a6bd;
            font-size: 13px;
            margin-bottom: 14px;
            line-height: 1.5;
        }
        .notice {
            border: 1px solid var(--border);
            background: rgba(13,27,46,.72);
            border-radius: 14px;
            padding: 14px 16px;
            color: #a9bad0;
            font-size: 13px;
            line-height: 1.6;
        }
        .result-card {
            padding: 27px;
            border-radius: 22px;
            border: 1px solid rgba(99,164,255,.22);
            background: linear-gradient(135deg, rgba(99,164,255,.12), rgba(13,27,46,.92));
            box-shadow: 0 18px 52px rgba(0,0,0,.22);
        }
        .result-kicker {
            color: #86bcff;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .1em;
            font-weight: 850;
        }
        .result-class { font-size: 46px; font-weight: 900; color: white; margin: 4px 0 0 0; }
        .result-confidence { color: #aabbd0; font-size: 14px; margin-top: 4px; }
        .class-card {
            border: 1px solid var(--border);
            background: rgba(13,27,46,.74);
            border-radius: 18px;
            padding: 18px;
            min-height: 150px;
        }
        .class-num { font-size: 13px; color: #86bcff; font-weight: 850; text-transform: uppercase; letter-spacing: .07em; }
        .class-title { color: white; font-size: 20px; font-weight: 850; margin-top: 5px; }
        .class-copy { color: #9aacc1; font-size: 12px; line-height: 1.5; margin-top: 8px; }
        .smallcaps { color: #89a2bf; font-size: 11px; letter-spacing: .1em; text-transform: uppercase; font-weight: 700; }
        .footer { border-top: 1px solid var(--border); margin-top: 42px; padding-top: 18px; color: #6f829a; font-size: 12px; }
        .stButton > button { border-radius: 12px; font-weight: 800; min-height: 44px; border: 1px solid rgba(99,164,255,.25); }
        div[data-testid="stMetric"] { background: rgba(13,27,46,.72); border: 1px solid var(--border); padding: 14px; border-radius: 16px; }
        [data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# DATABASE MODELS
# ============================================================================

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(default=True)

    students: Mapped[List["Student"]] = relationship(back_populates="user")


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    student_code: Mapped[Optional[str]] = mapped_column(String(80), nullable=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="students")
    predictions: Mapped[List["Prediction"]] = relationship(back_populates="student")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), index=True)
    predicted_class: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    class_0_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    class_1_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    class_2_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    class_3_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_inputs_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    student: Mapped[Student] = relationship(back_populates="predictions")
    features: Mapped[List["PredictionFeature"]] = relationship(back_populates="prediction")
    feedback: Mapped[List["Feedback"]] = relationship(back_populates="prediction")


class PredictionFeature(Base):
    __tablename__ = "prediction_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    feature_name: Mapped[str] = mapped_column(String(120))
    feature_value: Mapped[str] = mapped_column(String(255))

    prediction: Mapped[Prediction] = relationship(back_populates="features")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    actual_class: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prediction: Mapped[Prediction] = relationship(back_populates="feedback")


# ============================================================================
# DATABASE HELPERS
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_engine():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        DATABASE_URL,
        future=True,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return engine


@st.cache_resource(show_spinner=False)
def get_session_factory():
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def init_database() -> None:
    try:
        get_engine()
        get_session_factory()
    except Exception as exc:
        st.error("The local database could not be initialized. Please check the database folder permissions.")
        st.caption(str(exc))
        st.stop()


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


def authenticate_user(username: str, password: str) -> Optional[User]:
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        user = session.scalar(select(User).where(User.username == username.strip()))
        if user and user.is_active and verify_password(password, user.password_hash):
            return user
    return None


def register_user(username: str, email: str, password: str) -> Tuple[bool, str]:
    username = username.strip()
    email = email.strip().lower()
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        existing = session.scalar(
            select(User).where((User.username == username) | (User.email == email))
        )
        if existing:
            return False, "A user with that username or email already exists."
        user = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            is_active=True,
        )
        session.add(user)
        session.commit()
        return True, "Registration completed. You can now sign in."


def get_user_by_id(user_id: int) -> Optional[User]:
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        return session.get(User, user_id)


# ============================================================================
# MODEL / DATA LOADING
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "The saved model could not be loaded. This usually means the runtime's "
            "scikit-learn/joblib environment is incompatible with the version used to serialize "
            "the supplied model. Use the training environment's compatible scikit-learn version; "
            "do not retrain or replace the supplied model."
        ) from exc


@st.cache_resource(show_spinner=False)
def load_features() -> List[str]:
    if not FEATURES_PATH.exists():
        return MODEL_EXPECTED_FEATURES.copy()

    features = list(joblib.load(FEATURES_PATH))
    if features != MODEL_EXPECTED_FEATURES:
        raise ValueError(
            "feature_columns.pkl does not match the expected 18-feature model contract. "
            f"Loaded order: {features}"
        )
    return features


@st.cache_data(show_spinner=False)
def load_final_dataset() -> Optional[pd.DataFrame]:
    if not FINAL_DATA_PATH.exists():
        return None
    try:
        df = pd.read_csv(FINAL_DATA_PATH)
    except Exception as exc:
        raise RuntimeError(f"final_selected_features.csv could not be read: {exc}") from exc
    return df


@st.cache_data(show_spinner=False)
def load_raw_dataset() -> Optional[pd.DataFrame]:
    if not RAW_DATA_PATH.exists():
        return None
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except Exception as exc:
        raise RuntimeError(f"merged_dataset.csv could not be read: {exc}") from exc
    return df


init_database()

try:
    MODEL = load_model()
    FEATURE_COLUMNS = load_features()
    FINAL_DATASET = load_final_dataset()
    RAW_DATASET = load_raw_dataset()
except Exception as exc:
    st.error("EduPredict AI could not start with the supplied artifacts.")
    st.caption(str(exc))
    st.stop()


# ============================================================================
# FEATURE ENGINEERING AND VALIDATION
# ============================================================================

def validate_student_input(values: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required = set(RAW_INPUT_COLUMNS)
    missing = required - set(values)
    if missing:
        errors.append(f"Missing input fields: {sorted(missing)}")
        return errors

    checks = [
        ("StudyHours", 5, 44),
        ("Attendance", 60, 100),
        ("Age", 18, 29),
        ("OnlineCourses", 0, 20),
        ("AssignmentCompletion", 50, 100),
    ]
    for name, low, high in checks:
        try:
            value = float(values[name])
            if not low <= value <= high:
                errors.append(f"{name} must be between {low} and {high}.")
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric.")

    allowed = {
        "Resources": {0, 1, 2},
        "Extracurricular": {0, 1},
        "Motivation": {0, 1, 2},
        "Internet": {0, 1},
        "Gender": {0, 1},
        "LearningStyle": {0, 1, 2, 3},
        "Discussions": {0, 1},
        "EduTech": {0, 1},
        "StressLevel": {0, 1, 2},
    }
    for name, choices in allowed.items():
        try:
            value = int(values[name])
            if value not in choices:
                errors.append(f"{name} must be one of {sorted(choices)}.")
        except (TypeError, ValueError):
            errors.append(f"{name} must use one of {sorted(choices)}.")

    return errors


def engineer_features(raw_values: Dict[str, Any]) -> pd.DataFrame:
    errors = validate_student_input(raw_values)
    if errors:
        raise ValueError(" ".join(errors))

    x = {k: raw_values[k] for k in RAW_INPUT_COLUMNS}
    study_norm = float(x["StudyHours"]) / MAX_STUDY_HOURS
    attendance_norm = float(x["Attendance"]) / 100

    features = {
        "Academic_Engagement": (
            float(x["AssignmentCompletion"]) * 0.40
            + float(x["Attendance"]) * 0.30
            + study_norm * 100 * 0.20
            + int(x["Discussions"]) * 100 * 0.10
        ),
        "Assignment_Attendance_Index": float(x["AssignmentCompletion"]) * attendance_norm,
        "Study_Assignment_Index": study_norm * (float(x["AssignmentCompletion"]) / 100),
        "Study_Attendance_Index": study_norm * attendance_norm,
        "AssignmentCompletion": int(x["AssignmentCompletion"]),
        "Attendance_Normalized": attendance_norm,
        "Age": int(x["Age"]),
        "Attendance": int(x["Attendance"]),
        "StudyHours": int(x["StudyHours"]),
        "OnlineCourses": int(x["OnlineCourses"]),
        "StudyHours_Normalized": study_norm,
        "LearningStyle": int(x["LearningStyle"]),
        "Motivation_Study_Index": (int(x["Motivation"]) / MAX_MOTIVATION) * study_norm,
        "StressLevel": int(x["StressLevel"]),
        "Online_Learning_Activity": int(x["OnlineCourses"]) * (int(x["Internet"]) + int(x["EduTech"])),
        "Participation_Rate": (int(x["Extracurricular"]) + int(x["Discussions"])) / 2,
        "Digital_Learning_Index": (
            (int(x["OnlineCourses"]) / MAX_ONLINE_COURSES) * 0.40
            + int(x["Internet"]) * 0.20
            + int(x["EduTech"]) * 0.20
            + (int(x["Resources"]) / MAX_RESOURCES) * 0.20
        ),
        "Motivation": int(x["Motivation"]),
    }

    engineered = pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    return engineered


def get_class_probabilities(model: Any, data: pd.DataFrame) -> Tuple[List[int], Optional[np.ndarray]]:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(data))
        classes = [int(c) for c in model.classes_]
        return classes, probabilities
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(data))
        decision = np.atleast_2d(decision)
        decision = decision - decision.max(axis=1, keepdims=True)
        exp = np.exp(decision)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        classes = [int(c) for c in model.classes_]
        return classes, probabilities
    return [], None


def predict_student(raw_values: Dict[str, Any]) -> Dict[str, Any]:
    engineered = engineer_features(raw_values)
    prediction = int(MODEL.predict(engineered)[0])
    classes, probabilities = get_class_probabilities(MODEL, engineered)

    probability_map: Dict[int, float] = {}
    if probabilities is not None:
        for cls, prob in zip(classes, probabilities[0]):
            probability_map[int(cls)] = float(prob)
        confidence = max(probability_map.values()) * 100
    else:
        confidence = None

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_map,
        "engineered": engineered,
    }


def process_batch(batch: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    missing = [c for c in RAW_INPUT_COLUMNS if c not in batch.columns]
    if missing:
        raise ValueError(f"Missing required raw input columns: {missing}")

    output = batch.copy()
    predictions: List[Optional[int]] = []
    confidences: List[Optional[float]] = []
    class_probabilities = {f"Class_{c}_Probability": [] for c in range(4)}
    validation_errors: List[str] = []

    for _, row in batch.iterrows():
        raw = {c: row[c] for c in RAW_INPUT_COLUMNS}
        try:
            normalized = {
                "StudyHours": float(raw["StudyHours"]),
                "Attendance": float(raw["Attendance"]),
                "Resources": int(raw["Resources"]),
                "Extracurricular": int(raw["Extracurricular"]),
                "Motivation": int(raw["Motivation"]),
                "Internet": int(raw["Internet"]),
                "Gender": int(raw["Gender"]),
                "Age": int(raw["Age"]),
                "LearningStyle": int(raw["LearningStyle"]),
                "OnlineCourses": int(raw["OnlineCourses"]),
                "Discussions": int(raw["Discussions"]),
                "AssignmentCompletion": float(raw["AssignmentCompletion"]),
                "EduTech": int(raw["EduTech"]),
                "StressLevel": int(raw["StressLevel"]),
            }
            result = predict_student(normalized)
            predictions.append(result["prediction"])
            confidences.append(result["confidence"])
            validation_errors.append("")
            for c in range(4):
                class_probabilities[f"Class_{c}_Probability"].append(
                    result["probabilities"].get(c, np.nan) * 100
                )
        except Exception as exc:
            predictions.append(np.nan)
            confidences.append(np.nan)
            validation_errors.append(str(exc))
            for c in range(4):
                class_probabilities[f"Class_{c}_Probability"].append(np.nan)

    output["Predicted_FinalGrade"] = predictions
    output["Confidence"] = confidences
    for key, values in class_probabilities.items():
        output[key] = values
    output["Validation_Error"] = validation_errors
    return output, [e for e in validation_errors if e]


# ============================================================================
# DATA ANALYTICS HELPERS
# ============================================================================

@st.cache_data(show_spinner=False)
def get_class_profiles(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None or "FinalGrade" not in df.columns:
        return {
            "counts": pd.Series(dtype="int64"),
            "percentages": pd.Series(dtype="float64"),
            "means": pd.DataFrame(),
            "medians": pd.DataFrame(),
            "stds": pd.DataFrame(),
        }
    numeric = [c for c in df.columns if c != "FinalGrade" and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby("FinalGrade")[numeric]
    return {
        "counts": df["FinalGrade"].value_counts().sort_index(),
        "percentages": (df["FinalGrade"].value_counts(normalize=True).sort_index() * 100),
        "means": grouped.mean(),
        "medians": grouped.median(),
        "stds": grouped.std(),
    }


CLASS_PROFILES = get_class_profiles(FINAL_DATASET)


def historical_examscore_means() -> pd.Series:
    if RAW_DATASET is None or not {"FinalGrade", "ExamScore"}.issubset(RAW_DATASET.columns):
        return pd.Series(dtype="float64")
    return RAW_DATASET.groupby("FinalGrade")["ExamScore"].mean().sort_index()


def fmt_feature_name(name: str) -> str:
    return name.replace("_", " ")


def feature_display_value(feature: str, value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    if feature in {"Attendance", "AssignmentCompletion", "Age", "StudyHours", "OnlineCourses", "LearningStyle", "Motivation", "StressLevel"}:
        return f"{float(value):.0f}"
    if feature.endswith("_Normalized") or feature in {"Study_Attendance_Index", "Study_Assignment_Index", "Motivation_Study_Index", "Participation_Rate", "Digital_Learning_Index"}:
        return f"{float(value):.3f}"
    return f"{float(value):.2f}"


def build_feature_comparison(predicted_class: int, engineered: pd.DataFrame) -> pd.DataFrame:
    means = CLASS_PROFILES.get("means", pd.DataFrame())
    if means.empty or predicted_class not in means.index:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    student = engineered.iloc[0]
    class_means = means.loc[predicted_class]

    for feature in FEATURE_COLUMNS:
        if feature not in class_means.index:
            continue
        student_value = float(student[feature])
        class_mean = float(class_means[feature])
        difference = student_value - class_mean
        if np.isclose(class_mean, 0.0):
            diff_pct = np.nan
            status = "Above class average" if difference > 0 else "Below class average" if difference < 0 else "Near class average"
        else:
            diff_pct = (difference / abs(class_mean)) * 100
            if abs(diff_pct) <= 5:
                status = "Near class average"
            elif difference > 0:
                status = "Above class average"
            else:
                status = "Below class average"
        rows.append(
            {
                "Feature": feature,
                "Student Value": student_value,
                "Predicted Class Mean": class_mean,
                "Difference": difference,
                "Difference %": diff_pct,
                "Status": status,
            }
        )
    return pd.DataFrame(rows)


def generate_insights(comparison: pd.DataFrame) -> List[str]:
    if comparison.empty:
        return []

    priority = [
        "Attendance",
        "AssignmentCompletion",
        "StudyHours",
        "Study_Attendance_Index",
        "Study_Assignment_Index",
        "Academic_Engagement",
        "Digital_Learning_Index",
        "Participation_Rate",
    ]
    order = [f for f in priority if f in comparison["Feature"].tolist()]
    indexed = comparison.set_index("Feature")
    insights: List[str] = []
    for feature in order:
        row = indexed.loc[feature]
        if row["Status"] == "Above class average":
            insights.append(f"Your {fmt_feature_name(feature).lower()} is higher than the historical mean for the predicted class.")
        elif row["Status"] == "Below class average":
            insights.append(f"Your {fmt_feature_name(feature).lower()} is lower than the historical mean for the predicted class.")
        else:
            insights.append(f"Your {fmt_feature_name(feature).lower()} is near the historical mean for the predicted class.")
        if len(insights) >= 5:
            break
    return insights


def render_prediction_chart(probabilities: Dict[int, float]) -> go.Figure:
    plot_df = pd.DataFrame(
        {
            "Class": [f"Class {i}" for i in range(4)],
            "Probability": [probabilities.get(i, 0.0) * 100 for i in range(4)],
        }
    )
    fig = px.bar(plot_df, x="Class", y="Probability", text_auto=".2f")
    fig.update_layout(
        title="Class probability distribution",
        xaxis_title="FinalGrade class",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=360,
        margin=dict(l=15, r=15, t=60, b=25),
    )
    return fig


# ============================================================================
# DATABASE PERSISTENCE
# ============================================================================

def save_prediction(
    user_id: int,
    student_name: str,
    student_code: str,
    raw_inputs: Dict[str, Any],
    result: Dict[str, Any],
) -> int:
    SessionLocal = get_session_factory()
    probabilities = result["probabilities"]
    with SessionLocal() as session:
        student = Student(
            user_id=user_id,
            student_code=student_code.strip() or None,
            name=student_name.strip() or None,
            age=int(raw_inputs["Age"]),
            gender=int(raw_inputs["Gender"]),
        )
        session.add(student)
        session.flush()

        prediction = Prediction(
            user_id=user_id,
            student_id=student.id,
            predicted_class=int(result["prediction"]),
            confidence=(float(result["confidence"]) if result["confidence"] is not None else None),
            class_0_probability=probabilities.get(0, np.nan) * 100,
            class_1_probability=probabilities.get(1, np.nan) * 100,
            class_2_probability=probabilities.get(2, np.nan) * 100,
            class_3_probability=probabilities.get(3, np.nan) * 100,
            raw_inputs_json=json.dumps(raw_inputs),
        )
        session.add(prediction)
        session.flush()

        for feature in FEATURE_COLUMNS:
            value = result["engineered"].iloc[0][feature]
            feature_row = PredictionFeature(
                prediction_id=prediction.id,
                feature_name=feature,
                feature_value=str(float(value)) if isinstance(value, (np.floating, float)) else str(value),
            )
            session.add(feature_row)

        session.commit()
        return int(prediction.id)


def get_history(user_id: int) -> pd.DataFrame:
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        stmt = (
            select(Prediction, Student)
            .join(Student, Prediction.student_id == Student.id)
            .where(Prediction.user_id == user_id)
            .order_by(desc(Prediction.created_at))
        )
        rows = session.execute(stmt).all()
    records = []
    for prediction, student in rows:
        records.append(
            {
                "Prediction ID": prediction.id,
                "Student": student.name or student.student_code or f"Student #{student.id}",
                "Student Code": student.student_code or "",
                "Predicted Class": prediction.predicted_class,
                "Confidence": prediction.confidence,
                "Date": prediction.created_at,
            }
        )
    return pd.DataFrame(records)


def get_longitudinal_history(user_id: int) -> pd.DataFrame:
    """Return saved predictions with observable features for longitudinal analysis."""
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        stmt = (
            select(Prediction, Student, PredictionFeature)
            .join(Student, Prediction.student_id == Student.id)
            .join(PredictionFeature, PredictionFeature.prediction_id == Prediction.id)
            .where(Prediction.user_id == user_id)
            .order_by(Prediction.created_at.asc(), Prediction.id.asc())
        )
        rows = session.execute(stmt).all()

    records = []
    for prediction, student, feature in rows:
        student_label = student.student_code or student.name or f"Student #{student.id}"
        records.append({
            "Prediction ID": prediction.id,
            "Student ID": student.id,
            "Student": student_label,
            "Student Code": student.student_code or "",
            "Date": prediction.created_at,
            "Predicted Class": prediction.predicted_class,
            "Confidence": prediction.confidence,
            "Feature": feature.feature_name,
            "Value": pd.to_numeric(feature.feature_value, errors="coerce"),
        })

    if not records:
        return pd.DataFrame()

    long_df = pd.DataFrame(records)
    wide = long_df.pivot_table(
        index=["Prediction ID", "Student ID", "Student", "Student Code", "Date", "Predicted Class", "Confidence"],
        columns="Feature",
        values="Value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def _behavior_band(value: float, series: pd.Series) -> str:
    if pd.isna(value) or series.dropna().empty:
        return "Not enough data"
    q1, q2 = series.quantile([0.33, 0.67])
    if value <= q1:
        return "Lower in dataset"
    if value >= q2:
        return "Higher in dataset"
    return "Middle in dataset"


def build_plain_language_class_table() -> pd.DataFrame:
    means = CLASS_PROFILES.get("means", pd.DataFrame())
    if means.empty:
        return pd.DataFrame()

    rows = []
    source_df = FINAL_DATASET if FINAL_DATASET is not None else pd.DataFrame()
    for cls in range(4):
        if cls not in means.index:
            continue
        row = means.loc[cls]
        rows.append({
            "Class": f"Class {cls}",
            "Simple meaning": SUPPORT_PROFILES[cls]["title"],
            "Study effort": _behavior_band(row.get("StudyHours", np.nan), source_df.get("StudyHours", pd.Series(dtype=float))),
            "Attendance": _behavior_band(row.get("Attendance", np.nan), source_df.get("Attendance", pd.Series(dtype=float))),
            "Assignments": _behavior_band(row.get("AssignmentCompletion", np.nan), source_df.get("AssignmentCompletion", pd.Series(dtype=float))),
            "Engagement": _behavior_band(row.get("Academic_Engagement", np.nan), source_df.get("Academic_Engagement", pd.Series(dtype=float))),
            "Participation": _behavior_band(row.get("Participation_Rate", np.nan), source_df.get("Participation_Rate", pd.Series(dtype=float))),
            "Digital learning": _behavior_band(row.get("Digital_Learning_Index", np.nan), source_df.get("Digital_Learning_Index", pd.Series(dtype=float))),
            "Teacher support idea": SUPPORT_PROFILES[cls]["teacher_plan"],
        })
    return pd.DataFrame(rows)


def get_prediction_details(user_id: int, prediction_id: int) -> Optional[Dict[str, Any]]:
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        stmt = (
            select(Prediction, Student)
            .join(Student, Prediction.student_id == Student.id)
            .where(Prediction.id == prediction_id, Prediction.user_id == user_id)
        )
        result = session.execute(stmt).first()
        if not result:
            return None
        prediction, student = result
        features = session.scalars(
            select(PredictionFeature)
            .where(PredictionFeature.prediction_id == prediction_id)
            .order_by(PredictionFeature.id)
        ).all()
        feedback = session.scalars(
            select(Feedback)
            .where(Feedback.prediction_id == prediction_id)
            .order_by(desc(Feedback.created_at))
        ).all()

        raw_inputs = json.loads(prediction.raw_inputs_json or "{}")
        engineered = {item.feature_name: item.feature_value for item in features}
        probabilities = {
            0: prediction.class_0_probability,
            1: prediction.class_1_probability,
            2: prediction.class_2_probability,
            3: prediction.class_3_probability,
        }
        return {
            "prediction": prediction,
            "student": student,
            "raw_inputs": raw_inputs,
            "engineered": engineered,
            "probabilities": probabilities,
            "feedback": feedback,
        }


def save_feedback(user_id: int, prediction_id: int, actual_class: Optional[int], notes: str) -> Tuple[bool, str]:
    details = get_prediction_details(user_id, prediction_id)
    if details is None:
        return False, "Prediction not found."
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        item = Feedback(
            prediction_id=prediction_id,
            actual_class=actual_class,
            notes=notes.strip() or None,
        )
        session.add(item)
        session.commit()
    return True, "Feedback saved."


def dashboard_counts(user_id: int) -> Dict[str, int]:
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        total_students = int(session.scalar(select(func.count(Student.id)).where(Student.user_id == user_id)) or 0)
        total_predictions = int(session.scalar(select(func.count(Prediction.id)).where(Prediction.user_id == user_id)) or 0)
        counts = {
            "Total Students": total_students,
            "Total Predictions": total_predictions,
            "Class 0 Predictions": 0,
            "Class 1 Predictions": 0,
            "Class 2 Predictions": 0,
            "Class 3 Predictions": 0,
        }
        for cls in range(4):
            count = session.scalar(
                select(func.count(Prediction.id)).where(
                    Prediction.user_id == user_id,
                    Prediction.predicted_class == cls,
                )
            )
            counts[f"Class {cls} Predictions"] = int(count or 0)
        return counts


# ============================================================================
# UI HELPERS
# ============================================================================

def show_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">Machine Learning Product</div>
            <h1>🎓 EduPredict AI</h1>
            <p>Understand student performance through data-driven prediction. EduPredict AI uses a trained machine-learning model to classify students into one of four historical FinalGrade performance groups using study behavior, attendance, assignment completion, digital learning, participation, motivation, and related characteristics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_disclaimer() -> None:
    st.markdown(
        """
        <div class="notice">
        <b>Model disclaimer:</b> EduPredict AI provides a machine-learning prediction based on historical student data. The prediction is not a guaranteed academic result, official grade, institutional decision, or substitute for teacher/advisor judgment.<br><br>
        <b>Leakage control:</b> <code>ExamScore</code> is intentionally excluded from prediction because the system is designed as a pre-exam prediction workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_footer() -> None:
    st.markdown(
        '<div class="footer">EduPredict AI • Machine Learning Student Performance Prediction System • Built with Python, Streamlit and Scikit-learn.</div>',
        unsafe_allow_html=True,
    )


def select_coded_value(label: str, options: Sequence[int], descriptions: Dict[int, str], default: int = 0, help_text: Optional[str] = None) -> int:
    labels = [f"{value} — {descriptions.get(value, f'Category {value}')}" for value in options]
    default_idx = options.index(default) if default in options else 0
    selected = st.selectbox(label, labels, index=default_idx, help=help_text)
    return int(selected.split(" — ", 1)[0])


def raw_input_form(sample: Optional[Dict[str, Any]] = None, key_prefix: str = "predict") -> Tuple[Dict[str, Any], str, str, bool]:
    sample = sample or {}
    st.markdown('<div class="section-title">Student record</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Name and code are stored for auditability only. They are not model inputs.</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        student_name = st.text_input("Student Name (optional)", value=str(sample.get("student_name", "")), key=f"{key_prefix}_name")
    with m2:
        student_code = st.text_input("Student Code (optional)", value=str(sample.get("student_code", "")), key=f"{key_prefix}_code")

    st.markdown('<div class="section-title">A. Academic Activity</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        study_hours = st.number_input("Study Hours", min_value=5, max_value=44, value=int(sample.get("StudyHours", 20)), step=1, key=f"{key_prefix}_study_hours", help="Allowed range: 5–44.")
    with a2:
        attendance = st.number_input("Attendance (%)", min_value=60, max_value=100, value=int(sample.get("Attendance", 85)), step=1, key=f"{key_prefix}_attendance", help="Allowed range: 60–100.")
    with a3:
        assignment_completion = st.number_input("Assignment Completion (%)", min_value=50, max_value=100, value=int(sample.get("AssignmentCompletion", 85)), step=1, key=f"{key_prefix}_assignment", help="Allowed range: 50–100.")
    a4, a5 = st.columns(2)
    with a4:
        discussions = select_coded_value(
            "Discussions",
            [0, 1],
            {0: "No participation in discussions", 1: "Participates in discussions"},
            int(sample.get("Discussions", 1)),
            key_prefix,
        )
    with a5:
        extracurricular = select_coded_value(
            "Extracurricular",
            [0, 1],
            {0: "No extracurricular participation", 1: "Extracurricular participation"},
            int(sample.get("Extracurricular", 1)),
            key_prefix,
        )

    st.markdown('<div class="section-title">B. Digital Learning</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        online_courses = st.number_input("Online Courses", min_value=0, max_value=20, value=int(sample.get("OnlineCourses", 3)), step=1, key=f"{key_prefix}_courses")
    with d2:
        internet = select_coded_value(
            "Internet",
            [0, 1],
            {0: "No internet access", 1: "Internet access"},
            int(sample.get("Internet", 1)),
            key_prefix,
        )
    with d3:
        edutech = select_coded_value(
            "EduTech",
            [0, 1],
            {0: "Low/no educational technology usage", 1: "Educational technology available/used"},
            int(sample.get("EduTech", 1)),
            key_prefix,
        )
    with d4:
        resources = select_coded_value(
            "Resources",
            [0, 1, 2],
            {0: "Low resource availability", 1: "Medium resource availability", 2: "High resource availability"},
            int(sample.get("Resources", 1)),
            key_prefix,
        )

    st.markdown('<div class="section-title">C. Student Profile</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        age = st.number_input("Age", min_value=18, max_value=29, value=int(sample.get("Age", 20)), step=1, key=f"{key_prefix}_age")
    with p2:
        gender = select_coded_value(
            "Gender",
            [0, 1],
            {0: "Gender category 0", 1: "Gender category 1"},
            int(sample.get("Gender", 0)),
            key_prefix,
        )
    with p3:
        learning_style = select_coded_value(
            "Learning Style",
            [0, 1, 2, 3],
            {0: "Learning Style Category 0", 1: "Learning Style Category 1", 2: "Learning Style Category 2", 3: "Learning Style Category 3"},
            int(sample.get("LearningStyle", 2)),
            key_prefix,
        )

    st.markdown('<div class="section-title">D. Psychological / Behavioral</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        motivation = select_coded_value(
            "Motivation",
            [0, 1, 2],
            {0: "Low motivation category", 1: "Medium motivation category", 2: "High motivation category"},
            int(sample.get("Motivation", 2)),
            key_prefix,
        )
    with b2:
        stress_level = select_coded_value(
            "Stress Level",
            [0, 1, 2],
            {0: "Low stress category", 1: "Medium stress category", 2: "High stress category"},
            int(sample.get("StressLevel", 1)),
            key_prefix,
        )

    raw = {
        "StudyHours": int(study_hours),
        "Attendance": int(attendance),
        "Resources": int(resources),
        "Extracurricular": int(extracurricular),
        "Motivation": int(motivation),
        "Internet": int(internet),
        "Gender": int(gender),
        "Age": int(age),
        "LearningStyle": int(learning_style),
        "OnlineCourses": int(online_courses),
        "Discussions": int(discussions),
        "AssignmentCompletion": int(assignment_completion),
        "EduTech": int(edutech),
        "StressLevel": int(stress_level),
    }
    return raw, student_name, student_code, True


def sample_raw_student() -> Dict[str, Any]:
    return {
        "StudyHours": 20,
        "Attendance": 85,
        "Resources": 1,
        "Extracurricular": 1,
        "Motivation": 2,
        "Internet": 1,
        "Gender": 0,
        "Age": 20,
        "LearningStyle": 2,
        "OnlineCourses": 3,
        "Discussions": 1,
        "AssignmentCompletion": 85,
        "EduTech": 1,
        "StressLevel": 1,
    }


# ============================================================================
# AUTHENTICATION SCREEN
# ============================================================================

def auth_screen() -> None:
    show_header()
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="section-title">Welcome to EduPredict AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="notice">Sign in to use predictions, history, batch analysis, dataset exploration, and model insights. Passwords are stored only as secure password hashes.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-title">Workflow</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for col, (num, title, desc_text) in zip(
            cols,
            [
                ("01", "Predict", "Enter raw student information."),
                ("02", "Analyze", "View class probabilities and comparisons."),
                ("03", "Compare", "Benchmark against historical class means."),
                ("04", "Track", "Save and review predictions."),
            ],
        ):
            with col:
                st.markdown(f'<div class="smallcaps">{num}</div>', unsafe_allow_html=True)
                st.markdown(f"**{title}**")
                st.caption(desc_text)

    with right:
        tab_login, tab_register = st.tabs(["Login", "Register"])
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            if submitted:
                if not username or not password:
                    st.error("Enter both username and password.")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.user_id = user.id
                        st.session_state.username = user.username
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        with tab_register:
            with st.form("register_form"):
                username = st.text_input("Username", key="register_username")
                email = st.text_input("Email", key="register_email")
                password = st.text_input("Password", type="password", key="register_password")
                confirm = st.text_input("Confirm Password", type="password", key="register_confirm")
                submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                if len(username.strip()) < 3:
                    st.error("Username must contain at least 3 characters.")
                elif "@" not in email:
                    st.error("Enter a valid email address.")
                elif len(password) < 8:
                    st.error("Password must contain at least 8 characters.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, message = register_user(username, email, password)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)

    show_disclaimer()
    show_footer()


# ============================================================================
# PAGE RENDERERS
# ============================================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🎓 EduPredict AI")
        st.caption(f"Signed in as **{st.session_state.get('username', 'User')}**")
        page = st.radio(
            "Navigate",
            [
                "Dashboard",
                "Predict",
                "Prediction History",
                "Student Tracking",
                "Teacher Support Map",
                "Batch Prediction",
                "Class Profiles",
                "Model Insights",
                "Data Explorer",
                "Feature Guide",
                "About",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**Model contract**")
        st.markdown("- Target: `FinalGrade`")
        st.markdown("- Classes: `0, 1, 2, 3`")
        st.markdown("- Inputs: `14 raw fields → 18 model features`")
        if FINAL_DATASET is not None:
            st.markdown(f"- Training rows: `{len(FINAL_DATASET):,}`")
        st.markdown("---")
        st.markdown(
            '<div class="notice"><b>Leakage protection</b><br><code>ExamScore</code> and <code>FinalGrade</code> are never prediction inputs.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Log Out", use_container_width=True):
            for key in ["user_id", "username", "prediction_result", "prediction_saved_id"]:
                st.session_state.pop(key, None)
            st.rerun()
    return page


def render_dashboard(user_id: int) -> None:
    show_header()
    st.markdown('<div class="section-title">Dashboard</div>', unsafe_allow_html=True)
    counts = dashboard_counts(user_id)

    kpi_values = [
        ("Total Students", f"{counts['Total Students']:,}", "Students saved in this account"),
        ("Total Predictions", f"{counts['Total Predictions']:,}", "Persisted model predictions"),
        ("Class 0 Predictions", f"{counts['Class 0 Predictions']:,}", "Highest historical group"),
        ("Class 1 Predictions", f"{counts['Class 1 Predictions']:,}", "Upper-middle historical group"),
        ("Class 2 Predictions", f"{counts['Class 2 Predictions']:,}", "Lower-middle historical group"),
        ("Class 3 Predictions", f"{counts['Class 3 Predictions']:,}", "Lowest historical group"),
    ]
    row1 = st.columns(3)
    row2 = st.columns(3)
    for col, card in zip(row1 + row2, kpi_values):
        with col:
            label, value, sub = card
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Model at a glance</div>', unsafe_allow_html=True)
    mcols = st.columns(5)
    for col, (label, value) in zip(mcols, MODEL_METRICS.items()):
        with col:
            st.metric(label, f"{value:.2%}")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="section-title">Historical FinalGrade distribution</div>', unsafe_allow_html=True)
        if FINAL_DATASET is None:
            st.warning("Final selected dataset is not available.")
        else:
            counts_series = FINAL_DATASET["FinalGrade"].value_counts().sort_index()
            plot_df = pd.DataFrame({"Class": [f"Class {int(x)}" for x in counts_series.index], "Students": counts_series.values})
            fig = px.bar(plot_df, x="Class", y="Students", text_auto=True)
            fig.update_layout(height=380, margin=dict(l=15, r=15, t=45, b=25))
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Prediction history</div>', unsafe_allow_html=True)
        history = get_history(user_id)
        if history.empty:
            st.info("No predictions have been saved yet.")
        else:
            history_chart = history.copy()
            history_chart["Date"] = pd.to_datetime(history_chart["Date"]).dt.date
            daily = history_chart.groupby(["Date", "Predicted Class"]).size().reset_index(name="Predictions")
            fig = px.line(daily, x="Date", y="Predictions", color="Predicted Class", markers=True)
            fig.update_layout(height=380, margin=dict(l=15, r=15, t=45, b=25))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Recent predictions</div>', unsafe_allow_html=True)
    history = get_history(user_id)
    if history.empty:
        st.info("Your prediction history is empty.")
    else:
        st.dataframe(history.head(10), use_container_width=True, hide_index=True)
    show_disclaimer()


def render_predict_page(user_id: int) -> None:
    show_header()
    st.markdown('<div class="section-title">Single-student prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter meaningful raw student information. The application creates the exact 18 engineered features used by the saved model, in the exact saved order.</div>',
        unsafe_allow_html=True,
    )

    sample_col, _ = st.columns([1, 3])
    sample_clicked = False
    with sample_col:
        if st.button("Load Sample Student", use_container_width=True):
            st.session_state.predict_sample = sample_raw_student()
            sample_clicked = True
    if sample_clicked:
        st.rerun()

    sample = st.session_state.get("predict_sample", {})
    raw, student_name, student_code, _ = raw_input_form(sample=sample, key_prefix="single")

    st.markdown('<div class="section-title">Engineered feature preview</div>', unsafe_allow_html=True)
    try:
        preview = engineer_features(raw)
        pretty = preview.T.reset_index()
        pretty.columns = ["Feature", "Value"]
        pretty["Meaning"] = pretty["Feature"].map(FEATURE_HELP)
        st.dataframe(pretty, use_container_width=True, hide_index=True)
    except ValueError as exc:
        st.error(str(exc))
        preview = None

    action_col1, action_col2 = st.columns([1, 3])
    with action_col1:
        predict_clicked = st.button("Generate Prediction", type="primary", use_container_width=True)
    with action_col2:
        st.caption("ExamScore is never read or sent to the model.")

    if predict_clicked:
        try:
            errors = validate_student_input(raw)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                result = predict_student(raw)
                st.session_state.prediction_result = result
                st.session_state.prediction_raw = raw
                st.session_state.prediction_student_name = student_name
                st.session_state.prediction_student_code = student_code
                prediction_id = save_prediction(
                    user_id=user_id,
                    student_name=student_name,
                    student_code=student_code,
                    raw_inputs=raw,
                    result=result,
                )
                st.session_state.prediction_saved_id = prediction_id
                st.success(f"Prediction generated and saved to prediction history as #{prediction_id}.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("prediction_result")
    if not result:
        st.markdown('<div class="section-title">How the workflow works</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="notice"><b>Raw inputs</b> → validation → <b>exact feature engineering</b> → exact 18-column model DataFrame → saved preprocessing + model → FinalGrade class probabilities → historical class comparison → database storage.</div>',
            unsafe_allow_html=True,
        )
        show_disclaimer()
        return

    predicted = int(result["prediction"])
    confidence = result["confidence"]
    probs = result["probabilities"]
    comparison = build_feature_comparison(predicted, result["engineered"])

    st.markdown("---")
    left, right = st.columns([0.95, 1.25])
    with left:
        confidence_text = f"{confidence:.2f}%" if confidence is not None else "Not available"
        st.markdown(
            f'<div class="result-card"><div class="result-kicker">Predicted FinalGrade</div><div class="result-class">Class {predicted}</div><div class="result-confidence">Model confidence: {confidence_text}</div><div style="margin-top:14px;color:#93a8c1;font-size:13px;line-height:1.6">{CLASS_EXPLANATION[predicted]}</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="section-title">Class probability distribution</div>', unsafe_allow_html=True)
        for cls in range(4):
            pct = probs.get(cls, 0.0) * 100
            st.progress(min(max(pct / 100, 0), 1), text=f"Class {cls} — {pct:.2f}%")
        st.plotly_chart(render_prediction_chart(probs), use_container_width=True)

    st.markdown('<div class="section-title">What this class means</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="notice">{CLASS_EXPLANATION[predicted]}<br><br><b>Historical interpretation:</b> These class descriptions are derived from historical dataset patterns. They are not official institutional grade boundaries.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Student vs predicted-class historical mean</div>', unsafe_allow_html=True)
    if comparison.empty:
        st.info("Class comparison is not available because the final selected dataset is missing or does not contain the required class statistics.")
    else:
        display_comparison = comparison.copy()
        display_comparison["Student Value"] = display_comparison.apply(lambda r: feature_display_value(r["Feature"], r["Student Value"]), axis=1)
        display_comparison["Predicted Class Mean"] = display_comparison.apply(lambda r: feature_display_value(r["Feature"], r["Predicted Class Mean"]), axis=1)
        display_comparison["Difference"] = display_comparison["Difference"].map(lambda x: f"{x:+.4f}")
        display_comparison["Difference %"] = display_comparison["Difference %"].map(lambda x: "N/A" if pd.isna(x) else f"{x:+.2f}%")
        st.dataframe(display_comparison, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Data-driven interpretation</div>', unsafe_allow_html=True)
        insights = generate_insights(comparison)
        for insight in insights:
            st.write(f"• {insight}")
        st.caption("These statements describe differences from historical class means. They do not establish causation or guarantee future academic outcomes.")

    st.markdown('<div class="section-title">Closest alternative classes</div>', unsafe_allow_html=True)
    alternatives = sorted([(cls, prob) for cls, prob in probs.items() if cls != predicted], key=lambda x: x[1], reverse=True)
    alt_df = pd.DataFrame({"Class": [f"Class {c}" for c, _ in alternatives], "Probability": [p * 100 for _, p in alternatives]})
    st.dataframe(alt_df, use_container_width=True, hide_index=True)

    show_disclaimer()


def render_history_page(user_id: int) -> None:
    show_header()
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Database-backed predictions saved by your account.</div>', unsafe_allow_html=True)

    history = get_history(user_id)
    if history.empty:
        st.info("No predictions have been saved yet.")
        show_disclaimer()
        return

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        search = st.text_input("Search student")
    with f2:
        selected_class = st.selectbox("Predicted class", ["All", 0, 1, 2, 3])
    with f3:
        use_start_date = st.checkbox("Filter from date")
        start_date = st.date_input("From date", value=date.today()) if use_start_date else None
    with f4:
        use_end_date = st.checkbox("Filter to date")
        end_date = st.date_input("To date", value=date.today()) if use_end_date else None

    filtered = history.copy()
    if search:
        mask = filtered["Student"].str.contains(search, case=False, na=False) | filtered["Student Code"].str.contains(search, case=False, na=False)
        filtered = filtered[mask]
    if selected_class != "All":
        filtered = filtered[filtered["Predicted Class"] == int(selected_class)]
    if start_date:
        filtered = filtered[pd.to_datetime(filtered["Date"]).dt.date >= start_date]
    if end_date:
        filtered = filtered[pd.to_datetime(filtered["Date"]).dt.date <= end_date]

    sort_choice = st.selectbox("Sort", ["Newest first", "Confidence high to low", "Confidence low to high"])
    if sort_choice == "Confidence high to low":
        filtered = filtered.sort_values("Confidence", ascending=False)
    elif sort_choice == "Confidence low to high":
        filtered = filtered.sort_values("Confidence", ascending=True)
    else:
        filtered = filtered.sort_values("Date", ascending=False)

    shown = filtered.copy()
    shown["Confidence"] = shown["Confidence"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}%")
    st.dataframe(shown, use_container_width=True, hide_index=True)

    ids = filtered["Prediction ID"].tolist()
    if ids:
        chosen_id = st.selectbox("View prediction details", ids)
        details = get_prediction_details(user_id, int(chosen_id))
        if details:
            prediction = details["prediction"]
            student = details["student"]
            left, right = st.columns(2)
            with left:
                st.markdown('<div class="section-title">Student / raw inputs</div>', unsafe_allow_html=True)
                raw_df = pd.DataFrame(list(details["raw_inputs"].items()), columns=["Input", "Value"])
                raw_df["Meaning"] = raw_df["Input"].map({
                    "Resources": "Resource availability category",
                    "Extracurricular": "Extracurricular participation",
                    "Motivation": "Motivation category",
                    "Internet": "Internet access",
                    "Gender": "Gender category",
                    "LearningStyle": "Learning style category",
                    "Discussions": "Discussion participation",
                    "EduTech": "Educational technology usage/availability",
                    "StressLevel": "Stress category",
                }).fillna("Raw model workflow input")
                st.dataframe(raw_df, use_container_width=True, hide_index=True)
            with right:
                st.markdown('<div class="section-title">Class probabilities</div>', unsafe_allow_html=True)
                prob_df = pd.DataFrame(
                    {
                        "Class": [f"Class {c}" for c in range(4)],
                        "Probability": [details["probabilities"].get(c, np.nan) for c in range(4)],
                    }
                )
                prob_df["Probability"] = prob_df["Probability"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}%")
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                st.markdown(f"**Predicted:** Class {prediction.predicted_class}")
                st.markdown(f"**Confidence:** {prediction.confidence:.2f}%" if prediction.confidence is not None else "**Confidence:** N/A")

            st.markdown('<div class="section-title">Engineered features</div>', unsafe_allow_html=True)
            eng_df = pd.DataFrame(list(details["engineered"].items()), columns=["Feature", "Value"])
            eng_df["Meaning"] = eng_df["Feature"].map(FEATURE_HELP)
            st.dataframe(eng_df, use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">Feedback / actual result</div>', unsafe_allow_html=True)
            with st.form(f"feedback_{chosen_id}"):
                actual_options = ["Not available", 0, 1, 2, 3]
                actual_choice = st.selectbox("Actual FinalGrade", actual_options)
                notes = st.text_area("Notes")
                submitted = st.form_submit_button("Save Feedback")
            if submitted:
                actual = None if actual_choice == "Not available" else int(actual_choice)
                ok, message = save_feedback(user_id, int(chosen_id), actual, notes)
                (st.success if ok else st.error)(message)

            if details["feedback"]:
                feedback_df = pd.DataFrame(
                    [
                        {
                            "Actual Class": f.actual_class if f.actual_class is not None else "N/A",
                            "Notes": f.notes or "",
                            "Created": f.created_at,
                        }
                        for f in details["feedback"]
                    ]
                )
                st.dataframe(feedback_df, use_container_width=True, hide_index=True)

    show_disclaimer()



def render_student_tracking_page(user_id: int) -> None:
    show_header()
    st.markdown('<div class="section-title">Student Tracking</div>', unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Track how a student's saved predictions and observable study behaviors change over time. Use trends for support planning, not as a permanent label.</div>", unsafe_allow_html=True)

    df = get_longitudinal_history(user_id)
    if df.empty:
        st.info("No saved prediction history with feature values is available yet.")
        show_disclaimer()
        return

    students = sorted(df["Student"].dropna().unique().tolist())
    selected = st.selectbox("Student", students)
    student_df = df[df["Student"] == selected].sort_values("Date")
    latest = student_df.iloc[-1]
    cls = int(latest["Predicted Class"])

    st.markdown(
        f'<div class="result-card"><div class="result-kicker">Latest historical model view</div><div class="result-class">Class {cls}</div><div class="result-confidence">{SUPPORT_PROFILES[cls]["title"]}</div><div style="margin-top:14px;color:#93a8c1;font-size:13px;line-height:1.6">{SUPPORT_PROFILES[cls]["summary"]}</div></div>',
        unsafe_allow_html=True,
    )

    trend_features = [f for f in OBSERVABLE_FEATURES if f in student_df.columns]
    chart_feature = st.selectbox(
        "Trend feature",
        trend_features,
        format_func=lambda x: OBSERVABLE_FEATURE_LABELS.get(x, x),
    )
    trend = student_df[["Date", chart_feature]].copy()
    trend["Date"] = pd.to_datetime(trend["Date"])
    fig = px.line(
        trend,
        x="Date",
        y=chart_feature,
        markers=True,
        title=f"{OBSERVABLE_FEATURE_LABELS.get(chart_feature, chart_feature)} over time",
    )
    fig.update_layout(height=420, margin=dict(l=15, r=15, t=60, b=25))
    st.plotly_chart(fig, use_container_width=True)

    class_trend = student_df[["Date", "Predicted Class"]].copy()
    class_trend["Date"] = pd.to_datetime(class_trend["Date"])
    fig2 = px.line(class_trend, x="Date", y="Predicted Class", markers=True, title="Predicted class over saved history")
    fig2.update_yaxes(dtick=1)
    fig2.update_layout(height=360, margin=dict(l=15, r=15, t=60, b=25))
    st.plotly_chart(fig2, use_container_width=True)

    cols = st.columns(4)
    cols[0].metric("Saved observations", len(student_df))
    cols[1].metric("Latest class", f"Class {cls}")
    attendance = latest.get("Attendance", np.nan)
    assignments = latest.get("AssignmentCompletion", np.nan)
    cols[2].metric("Latest attendance", f"{attendance:.0f}%" if not pd.isna(attendance) else "N/A")
    cols[3].metric("Latest assignment completion", f"{assignments:.0f}%" if not pd.isna(assignments) else "N/A")

    st.markdown('<div class="section-title">Saved history</div>', unsafe_allow_html=True)
    display_cols = [c for c in ["Date", "Predicted Class", "Confidence"] + OBSERVABLE_FEATURES if c in student_df.columns]
    display = student_df[display_cols].copy()
    display["Date"] = pd.to_datetime(display["Date"]).dt.strftime("%Y-%m-%d %H:%M")
    if "Confidence" in display.columns:
        display["Confidence"] = display["Confidence"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}%")
    display = display.rename(columns=OBSERVABLE_FEATURE_LABELS)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Teacher support note</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="notice">{SUPPORT_PROFILES[cls]["teacher_plan"]}<br><br><b>Important:</b> attendance requirements should not be reduced because a student is predicted to perform well. Use this dashboard to decide where to add support, feedback, enrichment, or follow-up.</div>',
        unsafe_allow_html=True,
    )
    show_disclaimer()


def render_teacher_support_map(user_id: int) -> None:
    show_header()
    st.markdown('<div class="section-title">Teacher Support Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Group saved student observations by similarity in observable academic behavior. KMeans creates clusters; nearest-neighbor analysis shows the most similar saved observations.</div>', unsafe_allow_html=True)

    df = get_longitudinal_history(user_id)
    if df.empty:
        st.info("Run and save predictions first so the support map has student observations to analyze.")
        show_disclaimer()
        return

    feature_options = [f for f in OBSERVABLE_FEATURES if f in df.columns and df[f].notna().any()]
    st.markdown('<div class="section-title">Cluster settings</div>', unsafe_allow_html=True)
    selected_features = st.multiselect(
        "Features used for similarity",
        feature_options,
        default=[f for f in feature_options if f in {"StudyHours", "Attendance", "AssignmentCompletion", "Academic_Engagement", "Participation_Rate"}],
        format_func=lambda x: OBSERVABLE_FEATURE_LABELS.get(x, x),
    )

    if len(selected_features) < 2:
        st.warning("Choose at least two observable features to build the similarity map.")
        show_disclaimer()
        return

    work = df[["Prediction ID", "Student", "Date", "Predicted Class"] + selected_features].dropna(subset=selected_features).copy()
    if len(work) < 4:
        st.warning("At least four complete saved observations are needed for a four-cluster view.")
        show_disclaimer()
        return

    max_k = min(6, len(work))
    k = st.slider("Number of similarity clusters", min_value=2, max_value=max_k, value=min(4, max_k))

    X = work[selected_features].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    work["Similarity Cluster"] = model.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    work["PC1"] = coords[:, 0]
    work["PC2"] = coords[:, 1]

    st.markdown('<div class="section-title">Similarity clusters</div>', unsafe_allow_html=True)
    fig = px.scatter(
        work,
        x="PC1",
        y="PC2",
        color=work["Similarity Cluster"].astype(str),
        hover_data=["Student", "Predicted Class"] + selected_features,
        title="Student similarity map (PCA projection)",
    )
    fig.update_layout(height=620, xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"The first two principal components explain {pca.explained_variance_ratio_.sum():.1%} of the variance in the selected standardized features. Chart distance is a similarity approximation, not a grade score.")

    st.markdown('<div class="section-title">Cluster summaries</div>', unsafe_allow_html=True)
    cluster_summary = work.groupby("Similarity Cluster")[selected_features].mean().round(2)
    cluster_summary.insert(0, "Observations", work.groupby("Similarity Cluster").size())
    st.dataframe(cluster_summary, use_container_width=True)

    st.markdown('<div class="section-title">Nearest similar students</div>', unsafe_allow_html=True)
    nn = NearestNeighbors(n_neighbors=min(4, len(work)), metric="euclidean")
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    query_idx = st.selectbox(
        "Choose a saved observation",
        range(len(work)),
        format_func=lambda i: f"{work.iloc[i]['Student']} • prediction #{int(work.iloc[i]['Prediction ID'])}",
    )
    neighbor_rows = []
    for rank, (dist, idx) in enumerate(zip(distances[query_idx], indices[query_idx]), start=1):
        if idx == query_idx:
            continue
        neighbor_rows.append({
            "Rank": rank,
            "Student": work.iloc[idx]["Student"],
            "Prediction ID": int(work.iloc[idx]["Prediction ID"]),
            "Distance": float(dist),
            "Predicted Class": int(work.iloc[idx]["Predicted Class"]),
        })
    st.dataframe(pd.DataFrame(neighbor_rows).head(3), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">How teachers can use this</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="notice">Use clusters to spot groups with similar observable study patterns, then plan differentiated instruction: enrichment for students who are already consistent, standard support for stable groups, targeted intervention for groups with weaker observable habits, and closer follow-up for students showing multiple risk signals. The map should support teacher judgment, not replace it, and it should never be used to infer IQ or permanently label a student.</div>',
        unsafe_allow_html=True,
    )
    show_disclaimer()

def render_batch_page(user_id: int) -> None:
    show_header()
    st.markdown('<div class="section-title">Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Upload raw student-input CSV data. Do not upload the 18 engineered model features; the application creates them automatically.</div>', unsafe_allow_html=True)

    template = pd.DataFrame([sample_raw_student()])[RAW_INPUT_COLUMNS]
    st.download_button(
        "Download Raw-Input CSV Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="edupredict_raw_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        show_disclaimer()
        return

    try:
        batch = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Invalid CSV file: {exc}")
        return

    missing = [c for c in RAW_INPUT_COLUMNS if c not in batch.columns]
    if missing:
        st.error(f"Missing batch columns: {missing}")
        return

    extra = [c for c in batch.columns if c not in RAW_INPUT_COLUMNS]
    if extra:
        st.info(f"Extra columns detected: {extra}. They will be preserved in the output but ignored by the model.")

    st.markdown('<div class="section-title">Input preview</div>', unsafe_allow_html=True)
    st.dataframe(batch.head(100), use_container_width=True, hide_index=True)

    if st.button("Run Batch Prediction", type="primary"):
        try:
            output, errors = process_batch(batch)
            st.session_state.batch_output = output
            if errors:
                st.warning(f"{len(errors)} row(s) failed validation or prediction. See Validation_Error in the output.")
            else:
                st.success(f"Predicted {len(output):,} student row(s) successfully.")
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")

    output = st.session_state.get("batch_output")
    if output is not None:
        st.markdown('<div class="section-title">Batch results</div>', unsafe_allow_html=True)
        st.dataframe(output.head(200), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Predictions",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="edupredict_predictions.csv",
            mime="text/csv",
            type="primary",
        )

    show_disclaimer()


def render_class_profiles() -> None:
    show_header()
    st.markdown('<div class="section-title">Understand FinalGrade Classes</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Class labels are historical groups derived from the supplied dataset, not official institutional grade boundaries.</div>', unsafe_allow_html=True)

    counts = CLASS_PROFILES.get("counts")
    percentages = CLASS_PROFILES.get("percentages")
    if counts is None or counts.empty:
        st.warning("Class profile statistics are not available because final_selected_features.csv is missing.")
        return

    cards = st.columns(4)
    for cls, col in enumerate(cards):
        with col:
            count = int(counts.get(cls, 0))
            pct = float(percentages.get(cls, 0))
            st.markdown(
                f'<div class="class-card"><div class="class-num">Class {cls}</div><div class="class-title">{CLASS_LABELS[cls]}</div><div class="class-copy"><b>{count:,}</b> students • <b>{pct:.2f}%</b> of dataset</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Simple teacher-facing class language</div>', unsafe_allow_html=True)
    st.markdown('<div class="notice">The four classes are historical performance groups. They do not represent IQ, intelligence, personality, or “lazy/hardworking” labels. The behavior columns below are based on measurable features in the dataset.</div>', unsafe_allow_html=True)
    simple_table = build_plain_language_class_table()
    if not simple_table.empty:
        st.dataframe(simple_table, use_container_width=True, hide_index=True)

    exam_means = historical_examscore_means()
    st.markdown('<div class="section-title">Historical reference</div>', unsafe_allow_html=True)
    ref_df = pd.DataFrame(
        {
            "Class": [f"Class {c}" for c in range(4)],
            "Students": [int(counts.get(c, 0)) for c in range(4)],
            "Dataset %": [float(percentages.get(c, 0)) for c in range(4)],
            "Historical ExamScore Mean": [exam_means.get(c, np.nan) for c in range(4)],
        }
    )
    ref_df["Dataset %"] = ref_df["Dataset %"].map(lambda x: f"{x:.2f}%")
    ref_df["Historical ExamScore Mean"] = ref_df["Historical ExamScore Mean"].map(lambda x: "Not available" if pd.isna(x) else f"{x:.2f}")
    st.dataframe(ref_df, use_container_width=True, hide_index=True)
    st.caption("ExamScore is shown only as a historical after-exam reference from merged_dataset.csv. It is never used as a model input.")

    means = CLASS_PROFILES.get("means")
    medians = CLASS_PROFILES.get("medians")
    stds = CLASS_PROFILES.get("stds")
    if means is not None and not means.empty:
        st.markdown('<div class="section-title">Class mean profile — all selected numerical features</div>', unsafe_allow_html=True)
        mean_table = means.reindex(columns=FEATURE_COLUMNS).T.reset_index()
        mean_table.columns = ["Feature"] + [f"Class {int(c)} Mean" for c in means.index]
        st.dataframe(mean_table, use_container_width=True, hide_index=True)

        selected_class = st.selectbox("Inspect a class", [0, 1, 2, 3])
        detail_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Mean": [means.loc[selected_class, f] for f in FEATURE_COLUMNS],
                "Median": [medians.loc[selected_class, f] for f in FEATURE_COLUMNS],
                "Std Dev": [stds.loc[selected_class, f] for f in FEATURE_COLUMNS],
            }
        )
        st.markdown(f'<div class="section-title">Class {selected_class} statistics</div>', unsafe_allow_html=True)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        chart_df = means.reindex(columns=FEATURE_COLUMNS).T.copy()
        chart_df.index = [fmt_feature_name(x) for x in chart_df.index]
        melted = chart_df.reset_index(names="Feature").melt(id_vars="Feature", var_name="Class", value_name="Mean")
        fig = px.bar(melted, x="Feature", y="Mean", color="Class", barmode="group")
        fig.update_layout(height=560, xaxis_tickangle=-55, margin=dict(l=15, r=15, t=45, b=140))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Class definitions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="notice">These class descriptions are derived from historical dataset patterns. They are not official institutional grade boundaries.</div>',
        unsafe_allow_html=True,
    )
    show_disclaimer()


def render_model_insights() -> None:
    show_header()
    st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Model-evaluation results and feature-importance context from the supplied training notebook. These values describe model evaluation, not individual student predictions.</div>', unsafe_allow_html=True)

    c = st.columns(5)
    for col, (label, value) in zip(c, MODEL_METRICS.items()):
        with col:
            st.metric(label, f"{value:.2%}")

    st.markdown('<div class="section-title">Class-wise test-set metrics</div>', unsafe_allow_html=True)
    metrics_view = CLASS_METRICS.copy()
    for col in ["Precision", "Recall", "F1"]:
        metrics_view[col] = metrics_view[col].map(lambda x: f"{x:.4f}")
    st.dataframe(metrics_view, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Permutation feature importance</div>', unsafe_allow_html=True)
    imp = PERMUTATION_IMPORTANCE.sort_values("Importance_Mean", ascending=True)
    fig = px.bar(imp, x="Importance_Mean", y="Feature", orientation="h")
    fig.update_layout(height=700, xaxis_title="Mean permutation importance", yaxis_title="Feature")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(PERMUTATION_IMPORTANCE, use_container_width=True, hide_index=True)
    st.caption("Feature importance describes how strongly a feature contributed to model behavior; it does not prove causation.")

    st.markdown('<div class="section-title">Final model workflow</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="notice"><b>Model:</b> 12 Bagging<br><b>Estimator:</b> DecisionTreeClassifier inside BaggingClassifier<br><b>Estimators:</b> 100<br><b>Selection:</b> 20 candidate models → 5-fold cross-validation → top candidates → GridSearchCV → final untouched test-set evaluation.</div>',
        unsafe_allow_html=True,
    )
    show_disclaimer()


def render_data_explorer() -> None:
    show_header()
    st.markdown('<div class="section-title">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Explore dataset size, distributions, missingness, duplicates, class means, correlations, and feature-level charts using the supplied CSVs.</div>', unsafe_allow_html=True)

    if FINAL_DATASET is None and RAW_DATASET is None:
        st.warning("No dataset files are available.")
        return

    data_choice = st.radio("Dataset", ["Final selected features", "Original merged dataset"], horizontal=True)
    df = FINAL_DATASET if data_choice == "Final selected features" else RAW_DATASET
    if df is None:
        st.warning("Selected dataset is not available.")
        return

    a, b, c, d = st.columns(4)
    with a:
        st.metric("Dataset size", f"{len(df):,} rows")
    with b:
        st.metric("Columns", f"{df.shape[1]}")
    with c:
        st.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    with d:
        st.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    if "FinalGrade" in df.columns:
        st.markdown('<div class="section-title">FinalGrade distribution</div>', unsafe_allow_html=True)
        counts = df["FinalGrade"].value_counts().sort_index()
        dist = pd.DataFrame({"Class": [f"Class {int(x)}" for x in counts.index], "Students": counts.values})
        fig = px.pie(dist, names="Class", values="Students", hole=0.55)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Interactive feature explorer</div>', unsafe_allow_html=True)
    available_features = [c for c in df.columns if c != "FinalGrade"]
    selected_feature = st.selectbox("Feature", available_features)
    chart_type = st.selectbox("Chart type", ["Histogram", "Boxplot", "Class mean comparison", "Category distribution"])

    if chart_type == "Histogram":
        fig = px.histogram(df, x=selected_feature, nbins=30, marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Boxplot":
        if "FinalGrade" in df.columns:
            fig = px.box(df, x="FinalGrade", y=selected_feature, points=False)
        else:
            fig = px.box(df, y=selected_feature)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Class mean comparison":
        if "FinalGrade" not in df.columns:
            st.info("FinalGrade is not available for this dataset.")
        elif not pd.api.types.is_numeric_dtype(df[selected_feature]):
            st.info("Class mean comparison requires a numerical feature.")
        else:
            mean_df = df.groupby("FinalGrade")[selected_feature].mean().reset_index()
            fig = px.bar(mean_df, x="FinalGrade", y=selected_feature, text_auto=".2f")
            st.plotly_chart(fig, use_container_width=True)
    else:
        counts = df[selected_feature].value_counts(dropna=False).reset_index()
        counts.columns = [selected_feature, "Students"]
        fig = px.bar(counts, x=selected_feature, y="Students", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Numerical summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    if "FinalGrade" in df.columns:
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric) >= 2:
            st.markdown('<div class="section-title">Correlation heatmap</div>', unsafe_allow_html=True)
            corr = df[numeric].corr(numeric_only=True)
            fig = px.imshow(corr, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(200), use_container_width=True, hide_index=True)
    st.caption("ExamScore appears only in the original merged dataset for historical analytics. It is not part of the model's 18 prediction features.")
    show_disclaimer()


def render_feature_guide() -> None:
    show_header()
    st.markdown('<div class="section-title">Feature Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">The application accepts 14 raw fields and automatically creates the 18 features expected by the saved model.</div>', unsafe_allow_html=True)

    raw_table = pd.DataFrame(
        [
            ["StudyHours", "5–44", "Study-hours value in the training data."],
            ["Attendance", "60–100", "Student attendance percentage."],
            ["Resources", "0–2", "0 = low, 1 = medium, 2 = high resource availability."],
            ["Extracurricular", "0–1", "0 = no participation, 1 = participation."],
            ["Motivation", "0–2", "0 = low, 1 = medium, 2 = high motivation."],
            ["Internet", "0–1", "0 = no internet access, 1 = internet access."],
            ["Gender", "0–1", "Gender category 0 or gender category 1; source semantics are not invented."],
            ["Age", "18–29", "Student age."],
            ["LearningStyle", "0–3", "Learning Style Category 0–3; source data does not define semantic names."],
            ["OnlineCourses", "0–20", "Online-course activity count represented in the dataset."],
            ["Discussions", "0–1", "0 = no discussion participation, 1 = participates."],
            ["AssignmentCompletion", "50–100", "Percentage of assignments completed."],
            ["EduTech", "0–1", "0 = low/no educational technology usage, 1 = educational technology available/used."],
            ["StressLevel", "0–2", "0 = low, 1 = medium, 2 = high stress category."],
        ],
        columns=["Raw Input", "Allowed Range", "Meaning"],
    )
    st.dataframe(raw_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Engineered model features</div>', unsafe_allow_html=True)
    guide = pd.DataFrame(
        [[feature, FEATURE_DESCRIPTIONS.get(feature, "")] for feature in FEATURE_COLUMNS],
        columns=["Model Feature", "Explanation"],
    )
    st.dataframe(guide, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Exact feature-engineering formulas</div>', unsafe_allow_html=True)
    formulas = pd.DataFrame(
        [
            ["StudyHours_Normalized", "StudyHours / 44"],
            ["Attendance_Normalized", "Attendance / 100"],
            ["Study_Attendance_Index", "StudyHours_Normalized × Attendance_Normalized"],
            ["Academic_Engagement", "AssignmentCompletion × 0.40 + Attendance × 0.30 + StudyHours_Normalized × 100 × 0.20 + Discussions × 100 × 0.10"],
            ["Assignment_Attendance_Index", "AssignmentCompletion × Attendance_Normalized"],
            ["Study_Assignment_Index", "StudyHours_Normalized × (AssignmentCompletion / 100)"],
            ["Digital_Learning_Index", "(OnlineCourses / 20) × 0.40 + Internet × 0.20 + EduTech × 0.20 + (Resources / 2) × 0.20"],
            ["Online_Learning_Activity", "OnlineCourses × (Internet + EduTech)"],
            ["Motivation_Study_Index", "(Motivation / 2) × StudyHours_Normalized"],
            ["Participation_Rate", "(Extracurricular + Discussions) / 2"],
        ],
        columns=["Feature", "Formula"],
    )
    st.dataframe(formulas, use_container_width=True, hide_index=True)

    st.markdown('<div class="notice"><b>Target-leakage rule:</b> ExamScore and FinalGrade are never placed in the model input DataFrame.</div>', unsafe_allow_html=True)
    show_disclaimer()


def render_about() -> None:
    show_header()
    st.markdown('<div class="section-title">About EduPredict AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="notice"><b>Project objective:</b> provide a real ML product for predicting a student\'s historical FinalGrade performance group from pre-exam academic, behavioral, digital-learning, participation, motivation, and related characteristics.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Dataset & workflow</div>', unsafe_allow_html=True)
        st.write("The final selected dataset contains the 18 predictor features plus FinalGrade. The original merged dataset contains the raw student variables, ExamScore, and FinalGrade.")
        st.write("The notebook workflow included EDA, feature engineering, feature selection, model comparison, cross-validation, hyperparameter tuning, and final test evaluation.")
        st.write("ExamScore is intentionally excluded from prediction to avoid target leakage in a pre-exam workflow.")
    with right:
        st.markdown('<div class="section-title">Technology stack</div>', unsafe_allow_html=True)
        st.write("Python • Pandas • NumPy • Scikit-learn • Streamlit • Plotly • SQLAlchemy • SQLite • Joblib • Werkzeug password hashing")
        st.write("Model: BaggingClassifier using DecisionTreeClassifier base estimators, persisted as the supplied scikit-learn Pipeline.")

    st.markdown('<div class="section-title">Model evaluation</div>', unsafe_allow_html=True)
    metric_df = pd.DataFrame({"Metric": list(MODEL_METRICS.keys()), "Value": [f"{v:.4%}" for v in MODEL_METRICS.values()]})
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Class interpretation</div>', unsafe_allow_html=True)
    for cls in range(4):
        st.markdown(f"**Class {cls} — {SUPPORT_PROFILES[cls]['title']}**")
        st.caption(CLASS_EXPLANATION[cls])
        st.caption(SUPPORT_PROFILES[cls]["teacher_plan"])
    st.markdown('<div class="notice"><b>Why there is no IQ label:</b> the supplied data contains academic, behavioral, digital-learning, participation, motivation, stress, and related variables; it does not contain a validated IQ measure. Academic performance should not be treated as a proxy for IQ.</div>', unsafe_allow_html=True)

    show_disclaimer()


# ============================================================================
# MAIN APP
# ============================================================================

if "user_id" not in st.session_state:
    auth_screen()
else:
    page = render_sidebar()
    if page == "Dashboard":
        render_dashboard(int(st.session_state.user_id))
    elif page == "Predict":
        render_predict_page(int(st.session_state.user_id))
    elif page == "Prediction History":
        render_history_page(int(st.session_state.user_id))
    elif page == "Student Tracking":
        render_student_tracking_page(int(st.session_state.user_id))
    elif page == "Teacher Support Map":
        render_teacher_support_map(int(st.session_state.user_id))
    elif page == "Batch Prediction":
        render_batch_page(int(st.session_state.user_id))
    elif page == "Class Profiles":
        render_class_profiles()
    elif page == "Model Insights":
        render_model_insights()
    elif page == "Data Explorer":
        render_data_explorer()
    elif page == "Feature Guide":
        render_feature_guide()
    elif page == "About":
        render_about()
    show_footer()
