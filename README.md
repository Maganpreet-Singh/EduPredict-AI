# 🎓 EduPredict AI

> **Intelligent Student Performance & Academic Risk Prediction System**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Maganpreet-Singh/EduPredict-AI)

---

## 🚀 Overview

**EduPredict AI** is a machine-learning-powered academic analytics application designed to analyze student learning behavior and predict performance-related outcomes.

The project combines:

- 📊 Exploratory Data Analysis
- 🤖 Machine Learning classification
- 🎯 Probability-based prediction
- 🧠 Historical student tracking
- 🔍 Feature importance analysis
- 👥 Student grouping and clustering
- 📈 Interactive visual analytics
- 💾 Persistent prediction history
- 🧑‍🏫 Teacher-oriented academic support planning
- 🔐 User authentication and database-backed storage
- 🔄 What-if style student analysis

The goal is not simply to output a class label. The system is designed to turn academic data into **actionable insight** that can help educators identify patterns, understand contributing factors, monitor students over time, and plan appropriate academic support.

---

## ✨ Key Features

### 🎯 Student Performance Prediction

The application accepts student-related academic and behavioral inputs and produces a predicted performance class along with class probabilities.

The model uses features related to:

- Study effort
- Attendance
- Assignment completion
- Academic engagement
- Participation
- Online learning activity
- Motivation
- Stress level
- Learning style
- Digital learning usage
- Student age

### 📊 Prediction Probabilities

Instead of displaying only a single prediction, EduPredict AI provides probability estimates across the available performance classes.

This makes the prediction easier to interpret because users can see how strongly the model leans toward each class.

### 🧠 Historical Student Tracking

Predictions can be associated with students and stored in the application's database.

This creates the foundation for longitudinal academic analysis:

```
Student
   ↓
Prediction 1
   ↓
Prediction 2
   ↓
Prediction 3
   ↓
Performance trend
```

### 👥 Student Grouping & Clustering

EduPredict AI includes unsupervised-analysis components using:

- **K-Means clustering**
- **PCA**
- **Nearest Neighbors**

These techniques can be used to explore groups of students with similar observable academic patterns.

Students can be investigated according to combinations of:

- Attendance
- Study effort
- Assignment completion
- Academic engagement
- Participation
- Digital learning activity

> **Important:** The dataset does not contain a validated IQ measurement. Therefore, the system should not be interpreted as predicting or assigning IQ. Student groups should instead be understood through **observable academic behavior and historical performance patterns**.

### 🧑‍🏫 Teacher Support Profiles

The application translates model groups into teacher-oriented support descriptions.

Examples of support actions include:

- Regular academic check-ins
- Targeted feedback
- Structured practice
- Smaller learning milestones
- Frequent monitoring
- Focused intervention
- Independent enrichment work

The intention is to support teaching decisions, **not to label students permanently**.

### 🔎 Feature Importance

The project includes permutation-based feature importance analysis to investigate which model features contribute most strongly to prediction performance.

Examples of high-impact engineered features in the current analysis include:

- `Study_Assignment_Index`
- `Academic_Engagement`
- `Assignment_Attendance_Index`
- `Age`
- `Study_Attendance_Index`

### 📈 Interactive Analytics

The repository contains analytical and interactive outputs covering:

- Feature distributions
- Feature correlations
- Class distributions
- Study hours
- Attendance
- Assignment completion
- Motivation
- Stress level
- Learning style
- Online courses
- Participation
- Feature importance
- Confusion matrices
- Model comparison
- Class probabilities

---

## 🏗️ System Architecture

```
                 ┌──────────────────────┐
                 │   Student Inputs     │
                 └──────────┬───────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │ Feature Engineering    │
                │ & Preprocessing        │
                └────────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Trained ML Model      │
                 │ final_edupredict...   │
                 └───────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌─────────────┐
        │ Class    │   │ Probabili│   │ Feature     │
        │Prediction│   │ ties     │   │ Analysis    │
        └────┬─────┘   └────┬─────┘   └──────┬──────┘
             │              │                │
             └──────────────┬┴────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Historical Tracking  │
                 │ & Teacher Insights   │
                 └───────────┬──────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ SQLite Database │
                    └─────────────────┘
```

---

# 🧪 Machine Learning Pipeline

The project follows a structured ML workflow:

```
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Feature Selection
     ↓
Model Training
     ↓
Model Validation
     ↓
Hyperparameter Optimization
     ↓
Final Model
     ↓
Prediction + Probability Output
     ↓
Historical Monitoring
```

---

## 🧬 Feature Engineering

The project works with both original and engineered features.

| Feature | Purpose |
|---|---|
| `Academic_Engagement` | Combined academic participation and engagement representation |
| `Assignment_Attendance_Index` | Combines assignment completion and attendance |
| `Study_Assignment_Index` | Combines study effort and assignment completion |
| `Study_Attendance_Index` | Combines study effort and attendance |
| `Attendance_Normalized` | Normalized attendance representation |
| `StudyHours_Normalized` | Normalized study effort |
| `Motivation_Study_Index` | Combines motivation and study effort |
| `Online_Learning_Activity` | Represents online learning activity |
| `Participation_Rate` | Normalized participation indicator |
| `Digital_Learning_Index` | Represents digital learning access and usage |

---

# 📊 Model Evaluation

The current project analysis reports:

| Metric | Result |
|---|---:|
| Test Accuracy | **91.18%** |
| Macro Precision | **91.26%** |
| Macro Recall | **91.12%** |
| Macro F1 | **91.18%** |
| Cross-Validation Macro F1 | **86.39%** |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.9033 | 0.9206 | 0.9119 | 680 |
| 1 | 0.9077 | 0.9015 | 0.9046 | 589 |
| 2 | 0.9052 | 0.9193 | 0.9122 | 644 |
| 3 | 0.9342 | 0.9036 | 0.9186 | 581 |

> These are project-reported evaluation results stored in the repository artifacts. They should be interpreted in the context of the dataset, split strategy, preprocessing, and model-selection procedure used in the notebooks.

---

# 🔬 Model Analysis

The repository includes artifacts for comparing multiple candidate models and validating model performance.

```
artifacts/
├── 20_model_cross_validation.csv
├── 20_model_validation_comparison.csv
├── complete_20_model_comparison.csv
├── final_test_metrics.csv
├── finalgrade_classes.csv
├── gridsearch_results.csv
├── per_class_metrics.csv
├── permutation_feature_importance.csv
├── feature_columns.pkl
└── final_edupredict_model.pkl
```

These artifacts support inspection of:

- Cross-validation results
- Model comparisons
- Validation metrics
- Hyperparameter search
- Class-level metrics
- Feature importance
- Final model inputs
- Serialized final model

---

# 📁 Project Structure

```
EduPredict-AI/
│
├── 📂 Data/
│   ├── final_selected_features.csv
│   └── merged_dataset.csv
│
├── 📂 artifacts/
│   ├── final_edupredict_model.pkl
│   ├── feature_columns.pkl
│   ├── final_test_metrics.csv
│   ├── finalgrade_classes.csv
│   ├── per_class_metrics.csv
│   ├── permutation_feature_importance.csv
│   ├── gridsearch_results.csv
│   ├── 20_model_cross_validation.csv
│   ├── 20_model_validation_comparison.csv
│   └── complete_20_model_comparison.csv
│
├── 📂 database/
│   └── edupredict.db
│
├── 📂 image/
│   ├── model analysis plots
│   ├── class distribution plots
│   ├── feature analysis plots
│   ├── confusion matrices
│   ├── probability visualizations
│   └── interactive HTML visualizations
│
├── 📓 EDA.ipynb
├── 📓 Model.ipynb
├── 🐍 app.py
├── 📄 README.md
├── 📄 .gitignore
└── 📄 .gitattributes
```

---

# 🗂️ Dataset

The repository contains two main dataset files:

### `merged_dataset.csv`

Combined dataset used for the broader data-analysis workflow.

### `final_selected_features.csv`

Dataset containing the selected features used in the final modeling workflow.

The project uses observable student-related attributes such as:

```
StudyHours
Attendance
AssignmentCompletion
OnlineCourses
Motivation
StressLevel
LearningStyle
Participation
Digital Learning
Academic Engagement
```

The exact encoded representation of categorical variables should be interpreted according to the preprocessing and notebook implementation.

---

# 🖥️ Application

The main application is:

```
app.py
```

The application uses **Streamlit** as the interactive interface.

The code integrates:

- Streamlit UI
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib
- SQLAlchemy
- SQLite
- Werkzeug password hashing

---

# 💾 Database

EduPredict AI uses a SQLite database for application persistence.

The current implementation contains database entities for concepts including:

```
Users
Students
Predictions
Prediction Features
Feedback
```

This allows the system to associate users and students with prediction history and supporting feature data.

---

# 🔐 Authentication

The application includes user authentication with password hashing.

Credentials should never be hard-coded into the project.

For production deployment, environment variables and a production-grade database/authentication strategy should be preferred.

---

# 🛠️ Tech Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| ML | Scikit-learn, K-Means, PCA, Nearest Neighbors |
| Data Science | Pandas, NumPy |
| Visualization | Plotly |
| Web App | Streamlit |
| Database | SQLite, SQLAlchemy |
| Model Serialization | Joblib |
| Security | Werkzeug password hashing |

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Maganpreet-Singh/EduPredict-AI.git
```

Move into the project:

```bash
cd EduPredict-AI
```

Create a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

> **Note:** The repository currently contains the main application and model artifacts but does not expose a committed `requirements.txt` in the root listing checked during README preparation. Add one before distributing the project as a one-command installation package.

---

# 🧪 Running the Notebooks

### Exploratory Data Analysis

Open:

```
EDA.ipynb
```

The notebook covers the data-analysis workflow, including:

- Dataset inspection
- Cleaning
- Statistical exploration
- Feature distributions
- Correlation analysis
- Class/target analysis
- Visualization

### Model Development

Open:

```
Model.ipynb
```

The notebook covers:

- Feature engineering
- Feature selection
- Model training
- Model comparison
- Validation
- Cross-validation
- Hyperparameter tuning
- Final model evaluation
- Feature importance

---

# 📸 Visual Analysis

The repository contains a dedicated `image/` directory with generated analytical outputs.

Examples include:

```
numerical_feature_distributions.png
studyhours_distribution_by_finalgrade.png
study_assignment_index_distribution_by_finalgrade.png
study_attendance_index_distribution_by_finalgrade.png
learningstyle_distribution_by_finalgrade.png
motivation_distribution_by_finalgrade.png
stresslevel_distribution_by_finalgrade.png
participation_rate_distribution_by_finalgrade.png
permutation_feature_importance_-_12_bagging.png
normalized_confusion_matrix_-_12_bagging.png
student_finalgrade_class_probabilities.png
top_10_models_precision_recall_f1_comparison.png
```

Interactive HTML visualizations are also included for selected analyses.

---

# 🎯 Intended Use

EduPredict AI can be used as an academic analytics and decision-support project for exploring:

- Student performance patterns
- Academic engagement
- Attendance relationships
- Study behavior
- Assignment completion
- Digital learning activity
- Academic risk-oriented patterns
- Historical prediction trends
- Similar student groups

The system is intended to **support educators and students**, not replace human judgment.

---

# ⚠️ Responsible Use

Educational ML requires careful interpretation.

### Do not treat predictions as absolute truth

A prediction represents a statistical pattern learned from historical data. It is not a guaranteed future outcome.

### Do not use the system to label students permanently

Academic performance can change over time.

### Do not interpret model classes as intelligence or IQ

The current dataset does not provide a validated IQ measurement.

A model class should therefore be interpreted as a **historical performance pattern**, not as a measure of intelligence, ability, character, or potential.

### Use observable academic factors

Teacher interventions should be based on concrete indicators such as:

- Attendance
- Study effort
- Assignment completion
- Participation
- Motivation-related features
- Digital learning activity

The goal should be **appropriate academic support**, not punitive treatment.

---

# 🔮 Future Improvements

Potential improvements include:

- 📅 Semester-by-semester performance tracking
- 📉 Academic risk trend detection
- 📊 Student progress dashboards
- 🔔 Early-warning alerts
- 🧠 Explainable AI with SHAP
- 🔄 Automatic model retraining
- 🧪 Experiment tracking
- 🏫 Multi-class and multi-course support
- 📱 Mobile-friendly UI
- ☁️ Cloud deployment
- 👨‍🏫 Teacher dashboards
- 🎓 Student dashboards
- 📈 Longitudinal performance forecasting
- 🔐 Role-based access control
- 🗄️ PostgreSQL production database
- 🚀 CI/CD deployment
- 🐳 Docker containerization
- 📝 Automated data/model validation

---

# 🧭 Roadmap

```
✅ Data collection / merging
✅ EDA
✅ Feature engineering
✅ Feature selection
✅ Model comparison
✅ Cross-validation
✅ Hyperparameter tuning
✅ Final model
✅ Streamlit application
✅ Prediction probabilities
✅ Database-backed prediction history
✅ Student grouping / clustering
✅ Teacher support profiles

🔄 Better explainability
🔄 Longitudinal prediction analysis
🔄 Production deployment
🔄 Automated monitoring
🔄 Advanced early-warning system
```

---

# 📈 Why This Project Matters

Traditional academic systems often answer:

> **“What grade did the student get?”**

EduPredict AI explores a broader question:

> **“What patterns are associated with the student's current academic performance, and how can those patterns inform timely support?”**

The workflow is:

```
Data
 ↓
Patterns
 ↓
Prediction
 ↓
Interpretation
 ↓
Historical Tracking
 ↓
Academic Support
```

---

# 👨‍💻 Author

**Maganpreet Singh**

B.Tech — Computer Science & Engineering

GitHub: [Maganpreet-Singh](https://github.com/Maganpreet-Singh)

Project: [EduPredict-AI](https://github.com/Maganpreet-Singh/EduPredict-AI)

---

# ⭐ Support the Project

If you find this project useful for learning, experimentation, or academic analytics, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

A license file is not currently included in the repository.

Add a license before treating the project as an open-source project for external reuse.

---

> **EduPredict AI — Turning student data into meaningful academic insight.** 🎓📊🤖
