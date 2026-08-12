# Disease_Prediction


# 🏥 Healthcare Predictive Analytics — Multi-Disease Detection

A machine learning project that predicts **possible diseases based on reported symptoms** using a supervised multiclass classification model. The project uses a **Mendeley Data** and **Logistic Regression** to classify symptoms into multiple disease categories.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes only. It is **not a medical diagnostic tool** and should not be used as a substitute for professional medical advice.

---

## 📌 Project Overview

Healthcare professionals often need to consider multiple symptoms when identifying possible diseases. This project explores how machine learning can assist with **symptom-based disease prediction**.

The system takes a set of symptoms as input and predicts the most likely disease category learned from the training dataset.

### Main workflow

```text
Patient Symptoms
       ↓
Data Preprocessing
       ↓
Disease Filtering
       ↓
Train/Test Split
       ↓
Logistic Regression
       ↓
Disease Prediction
       ↓
Model Evaluation
```

---

## 🎯 Objectives

* Build a machine learning model for **multi-disease prediction**.
* Use symptoms as predictive features.
* Clean and preprocess healthcare-related data.
* Handle diseases with very few training examples.
* Train a **Logistic Regression** classifier.
* Evaluate the model using:

  * Accuracy
  * Macro F1 Score
  * Weighted F1 Score
* Analyze important symptoms/features.
* Provide a foundation for a future healthcare prediction application.

---

## 📊 Dataset

The project uses a **disease and symptoms dataset obtained from Kaggle**.

The original dataset contained approximately:

* **189,000+ records**
* **700+ disease classes**
* **377 symptom features**

The target variable is:

```text
diseases
```

The remaining columns represent symptoms.

### Data representation

Symptoms are represented as numerical/binary features, allowing the machine learning model to learn relationships between symptoms and disease classes.

Example:

| fever | headache | vomiting | cough | Disease   |
| ----: | -------: | -------: | ----: | --------- |
|     1 |        1 |        0 |     0 | Disease A |
|     0 |        1 |        1 |     0 | Disease B |
|     1 |        0 |        0 |     1 | Disease C |

---

## 🧹 Data Preprocessing

Several preprocessing steps were performed before model training.

### 1. Duplicate removal

Duplicate records were removed to reduce repeated observations.

### 2. Rare disease filtering

The dataset contained many diseases with very few examples.

The final experiment retained diseases with **at least 20 records**.

This reduced the number of disease classes from:

```text
727 → 512 diseases
```

This was done because a classification model cannot reliably learn a disease from only a handful of examples.

### 3. Feature and target separation

```text
X → Symptoms
y → Disease
```

### 4. Train/Test split

The dataset was divided into:

```text
80% → Training
20% → Testing
```

A stratified split was used to maintain the distribution of disease classes between the training and testing sets.

---

# 🤖 Machine Learning Model

## Logistic Regression

The final model uses **Logistic Regression** for multiclass classification.

Although Logistic Regression is commonly associated with binary classification, it can also be used to classify multiple categories.

In this project:

```text
Input:
Multiple symptoms

Output:
Predicted disease
```

### Model configuration

```python
LogisticRegression(
    max_iter=200,
    solver="lbfgs",
    multi_class="multinomial",
    n_jobs=2
)
```

---

# 📈 Model Performance

The final Logistic Regression model was evaluated using the test dataset.

### Results

| Metric                |      Score |
| --------------------- | ---------: |
| **Accuracy**          | **83.09%** |
| **Macro F1 Score**    | **0.6948** |
| **Weighted F1 Score** | **0.8302** |

### Interpretation

**Accuracy — 83.09%**

The model correctly classified approximately 83% of the test observations.

**Macro F1 — 0.6948**

Macro F1 calculates the F1 score independently for each disease and then gives each disease equal importance. This is particularly useful because the dataset contains diseases with different numbers of examples.

**Weighted F1 — 0.8302**

Weighted F1 accounts for the number of samples belonging to each disease class.

---

# 🔬 Model Comparison

Several approaches were explored during development.

| Model                           | Diseases |   Accuracy |   Macro F1 | Weighted F1 |
| ------------------------------- | -------: | ---------: | ---------: | ----------: |
| Logistic Regression — Initial   |      727 |     80.48% |     0.5564 |      0.7944 |
| **Logistic Regression — Final** |  **512** | **83.09%** | **0.6948** |  **0.8302** |
| Linear SVM                      |      512 |     80.57% |     0.5433 |      0.8044 |

Based on these experiments, the **final Logistic Regression model** performed best among the tested approaches.

---

# ⭐ Feature Importance

Feature importance analysis is performed using the coefficients learned by Logistic Regression.

For the multiclass model, the absolute coefficients are aggregated across disease classes to obtain an overall importance score for each symptom.

This allows us to identify symptoms that have a stronger influence on the model's predictions.

Example analysis:

```text
Symptoms
   ↓
Logistic Regression Coefficients
   ↓
Absolute coefficient values
   ↓
Average across disease classes
   ↓
Global symptom importance
```

The project also generates a visualization of the **Top 20 Important Symptoms**.

---

# 🗂️ Project Structure

```text
Healthcare-Predictive-Analytics/
│
├── dataset/
│   └── Disease and symptoms dataset.csv
│
├── notebooks/
│   └── disease_prediction.ipynb
│
├── models/
│   ├── disease_prediction_logistic_regression.pkl
│   └── symptom_columns.pkl
│
├── visualizations/
│   └── feature_importance.png
│
├── README.md
│
└── requirements.txt
```

---

# 💾 Saved Model Files

After training, the model was saved using `joblib`.

### Model

```text
disease_prediction_logistic_regression.pkl
```

Contains the trained Logistic Regression model.

### Symptom columns

```text
symptom_columns.pkl
```

Contains the exact symptom feature names used during training.

These files can later be loaded into a web application such as **Streamlit**.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Joblib**
* **Google Colab**
* **Kaggle Dataset**

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/healthcare-predictive-analytics.git
```

Move into the project directory:

```bash
cd healthcare-predictive-analytics
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

---

# 🚀 Running the Project

Open the notebook:

```text
notebooks/disease_prediction.ipynb
```

Run the cells sequentially:

```text
1. Import libraries
2. Load dataset
3. Explore dataset
4. Clean data
5. Remove duplicate records
6. Filter rare diseases
7. Separate features and target
8. Train/test split
9. Train Logistic Regression
10. Evaluate model
11. Calculate F1 scores
12. Perform feature importance analysis
13. Save trained model
```

---

# 🔮 Future Improvements

The current project provides a strong machine learning foundation, but several improvements can be made.

### 1. Web Application

Build an interactive **Streamlit** application where users can select symptoms and receive model predictions.

### 2. Top-5 Predictions

Instead of returning only one disease, the application can display the top five model predictions.

### 3. Explainable AI

Add explanations showing which selected symptoms contributed most strongly to a prediction.

### 4. More Medical Data

Integrate larger and more clinically validated datasets.

### 5. Model Optimization

Experiment with additional classification techniques and hyperparameter optimization.

### 6. Better Class Balancing

Investigate techniques such as class weighting or resampling to improve performance on less-represented diseases.

---

# 🔐 Ethical Considerations & Privacy

Healthcare data requires careful handling.

This project follows the following principles:

* Do not collect personally identifiable information.
* Do not store patient names, addresses, phone numbers, or medical IDs.
* Use anonymized/public datasets for experimentation.
* Do not expose individual patient records.
* Clearly communicate model limitations.
* Do not present predictions as confirmed medical diagnoses.

### Important limitation

The model learns patterns from the dataset used for training. Its predictions may not generalize to every population, hospital, age group, or clinical situation.

Therefore:

> **A model prediction should never be treated as a confirmed diagnosis. A qualified healthcare professional should make the final medical assessment.**

---

# 📚 Learning Outcomes

Through this project, the following concepts were implemented:

* Data preprocessing
* Exploratory data analysis
* Multiclass classification
* Logistic Regression
* Train/Test splitting
* Stratified sampling
* Model evaluation
* Accuracy
* Precision
* Recall
* F1 Score
* Macro F1
* Weighted F1
* Feature importance
* Model serialization using Joblib
* Healthcare AI ethics
* Data privacy

---

# 👩‍💻 Author

**Sneha Singh**

**Project:** Healthcare Predictive Analytics – Multi-Disease Detection

---

## ⭐ Project Summary

> **Healthcare Predictive Analytics is a machine learning project that uses symptom-based data to predict multiple disease categories. After preprocessing and filtering extremely rare disease classes, a multinomial Logistic Regression model was trained on 512 disease classes and achieved 83.09% accuracy, 0.6948 Macro F1, and 0.8302 Weighted F1 on the test set. The project also incorporates feature importance analysis and emphasizes responsible use, data privacy, and the limitations of AI-based medical prediction.**
