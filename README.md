# ❤️ CardioPredict

### Heart Disease Prediction using Machine Learning

CardioPredict is an end-to-end machine learning project that predicts the presence of heart disease from clinical parameters.

The project uses a **Gradient Boosting Classifier** trained with a leakage-safe preprocessing pipeline and provides an interactive **Streamlit web application** for making predictions.

> ⚠️ **Disclaimer:** This project is for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used for clinical decision-making.

---

## 🚀 Features

- Binary heart disease classification
- Data cleaning and preprocessing
- Missing-value handling
- Categorical feature encoding
- Numerical feature scaling
- Stratified train/test split
- Comparison of multiple machine learning models
- Hyperparameter tuning using GridSearchCV
- ROC-AUC based model evaluation
- Feature importance analysis
- Interactive Streamlit web application
- Dark / light mode
- Patient-level prediction with probability output
- Saved trained model using Joblib

---

## 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Target Creation
   ↓
Train / Test Split
   ↓
Preprocessing Pipeline
   ├── Missing Value Imputation
   ├── Feature Scaling
   └── One-Hot Encoding
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Final Gradient Boosting Model
   ↓
Evaluation
   ↓
Model Serialization
   ↓
Streamlit Application
```

---

## 📊 Dataset

The project uses the `heart_disease_uci.csv` dataset.

### Dataset shape

```text
920 rows × 16 columns
```

The dataset contains clinical and demographic features such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- ST depression
- ST segment slope
- Number of major vessels
- Thalassemia

The original `num` target was converted into a binary classification target:

```text
target = 0 → No heart disease
target = 1 → Heart disease
```

using:

```python
target = (num > 0).astype(int)
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline handles both numerical and categorical features.

### Numerical features

```text
age
trestbps
chol
thalch
oldpeak
ca
```

Processing:

```text
Missing values
      ↓
Median imputation
      ↓
StandardScaler
```

### Categorical features

```text
sex
cp
fbs
restecg
exang
slope
thal
```

Processing:

```text
Missing values
      ↓
Most-frequent imputation
      ↓
One-Hot Encoding
```

The preprocessing steps are included inside the machine learning pipeline to prevent data leakage.

---

## 🤖 Models Compared

Four classification algorithms were evaluated:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.98% | 81.90% | 84.31% | 83.09% | 0.890 |
| Random Forest | 79.35% | 79.63% | 84.31% | 81.90% | 0.900 |
| Gradient Boosting | 82.07% | 80.53% | 89.22% | 84.65% | 0.906 |
| SVM | 81.52% | 80.36% | 88.24% | 84.11% | 0.898 |

---

## 🎯 Final Model

The final model is a **Gradient Boosting Classifier**.

Hyperparameters were tuned using:

```text
GridSearchCV
5-fold cross-validation
ROC-AUC scoring
```

### Final test performance

| Metric | Result |
|---|---:|
| Accuracy | **82.61%** |
| Precision | **82.41%** |
| Recall | **87.25%** |
| F1 Score | **84.76%** |
| ROC-AUC | **0.909** |

The final model was evaluated on a held-out test set that was not used during model training.

> **Important:** ROC-AUC of `0.909` is not the same as 90.9% accuracy. The test accuracy is **82.61%**.

---

## 🖥️ Streamlit Application

CardioPredict includes an interactive Streamlit interface where users can enter clinical parameters and receive:

- Model prediction
- Estimated probability
- Model performance information
- Dark / light interface

### Application flow

```text
Enter Patient Information
          ↓
Submit Prediction
          ↓
Trained ML Pipeline
          ↓
Prediction + Probability
```

---

## 📁 Project Structure

```text
heart-disease-ml/
│
├── heart_disease_uci.csv
├── heart_disease_ml.ipynb
├── heart_disease_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Joblib

### Visualization

- Matplotlib
- Seaborn

### Frontend / Deployment

- Streamlit

### Development

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## 🔧 Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd heart-disease-ml
```

Create a virtual environment:

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

---

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Reproduce the Model

To retrain the model:

1. Open:

```text
heart_disease_ml.ipynb
```

2. Run the notebook from top to bottom.

3. The trained model will be saved as:

```text
heart_disease_model.pkl
```

4. Start the Streamlit application:

```bash
streamlit run app.py
```

---

## 🔐 Leakage Prevention

The project uses a proper train/test workflow.

The dataset is split before fitting preprocessing transformations:

```text
Raw Dataset
     ↓
Train/Test Split
     ↓
Training Data → Fit preprocessing
     ↓
Test Data → Transform using fitted preprocessing
```

This prevents information from the test set from influencing preprocessing during training.

The `dataset` source column and original `id` were excluded from the modeling features to avoid learning source-specific or identifier-related patterns.

---

## 📈 Model Evaluation

The project evaluates models using multiple metrics rather than relying only on accuracy.

### Accuracy

Measures the proportion of correctly classified samples.

### Precision

Measures how many samples predicted as positive were actually positive.

### Recall

Measures how many actual positive samples were detected.

### F1 Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

---

## 💾 Model Serialization

The complete trained preprocessing + model pipeline is saved using Joblib:

```python
joblib.dump(best_model, "heart_disease_model.pkl")
```

This allows the Streamlit application to load the already-trained pipeline without retraining the model every time the application starts.

---

## ⚠️ Limitations

This project is an educational machine learning implementation and has several limitations:

- The dataset is relatively small.
- Performance depends on the characteristics of the dataset.
- The model has not been clinically validated.
- Test-set performance does not guarantee real-world clinical performance.
- Model predictions should not be interpreted as medical diagnoses.
- Additional external validation would be required before any real clinical application.

---

## 🎓 Learning Outcomes

Through this project, the following machine learning concepts were practiced:

- Exploratory Data Analysis
- Data cleaning
- Missing-value treatment
- Feature preprocessing
- Categorical encoding
- Feature scaling
- Train/test splitting
- Classification
- Model comparison
- Hyperparameter tuning
- Cross-validation
- ROC-AUC evaluation
- Feature importance
- Model serialization
- Streamlit deployment

---

## 👨‍💻 Project

**CardioPredict — Heart Disease Prediction using Machine Learning**

Built as an educational machine learning project to demonstrate an end-to-end ML workflow from raw data to an interactive application.
