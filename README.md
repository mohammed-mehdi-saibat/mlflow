# Customer Churn Prediction with MLflow

A machine learning live coding project demonstrating how to build, evaluate, compare, and track customer churn classification models using **Scikit-learn** and **MLflow**.

The project also investigates whether adding a previously generated **customer cluster** improves churn prediction.

## 🎯 Objective

The main objective is to build a classification pipeline that predicts whether a customer will churn.

Two questions are explored:

1. Which classification model performs best?
2. Does adding the `cluster` feature improve churn prediction?

Each model is therefore trained and evaluated in two configurations:

* **Without cluster**
* **With cluster**

---

## 🔬 Machine Learning Workflow

```text
Cleaned Customer Data
        │
        ▼
Train / Test Split
        │
        ▼
Preprocessing
   ┌────┴─────┐
   ▼          ▼
Numerical   Categorical
   │          │
   ▼          ▼
Standard    One-Hot
Scaler      Encoding
   └────┬─────┘
        ▼
     Pipeline
        │
        ▼
 Classification
    ┌───┴────┐
    ▼        ▼
Logistic   Random
Regression Forest
    │        │
    └───┬────┘
        ▼
   Evaluation
        │
        ▼
      MLflow
```

---

## 🤖 Models

The project compares:

### Logistic Regression

A linear classification model used as a strong and interpretable baseline.

### Random Forest

An ensemble of decision trees capable of learning non-linear relationships between customer characteristics and churn.

Each model is evaluated:

```text
Model                  Cluster
--------------------------------
Logistic Regression    Without
Logistic Regression    With
Random Forest          Without
Random Forest          With
```

This results in **4 MLflow runs**.

---

## ⚙️ Preprocessing

The preprocessing is handled using Scikit-learn's `ColumnTransformer`.

### Numerical features

Numerical features are standardized using:

```python
StandardScaler()
```

This puts numerical variables on comparable scales.

### Categorical features

Categorical variables are encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
```

`handle_unknown="ignore"` prevents the pipeline from failing if a category appears during testing or prediction that was not present during training.

### Complete Pipeline

Preprocessing and the classifier are combined into a single pipeline:

```python
Pipeline([
    ("prep", preprocessor),
    ("model", model)
])
```

This ensures that the same preprocessing steps are automatically applied during both training and prediction.

---

## 📊 Evaluation Metrics

The models are evaluated using:

### Precision

Measures how many customers predicted as churners actually churned.

### Recall

Measures how many of the customers who actually churned were correctly identified.

### F1-score

Provides a balance between precision and recall.

### ROC-AUC

Measures how well the model separates churners from non-churners across different classification thresholds.

The probability of the positive class is obtained with:

```python
y_proba = pipe.predict_proba(X_test)[:, 1]
```

Here:

```text
class 0 → no churn
class 1 → churn
```

---

## 📈 MLflow Tracking

MLflow is used to track each experiment.

For every run, the project logs:

### Parameters

* Model type
* Whether `cluster` is used
* Class weight

### Metrics

* Precision
* Recall
* F1-score
* ROC-AUC

### Model

The complete Scikit-learn pipeline is logged:

```python
mlflow.sklearn.log_model(
    pipe,
    name="model",
    serialization_format="cloudpickle"
)
```

This allows the preprocessing and trained model to be stored together.

---

## 🧪 Experiment Tracking

The experiment generates runs such as:

```text
logistic_regression_without_cluster
logistic_regression_with_cluster
random_forest_without_cluster
random_forest_with_cluster
```

The MLflow UI can then be used to compare:

```text
             Without Cluster     With Cluster
                  │                  │
                  ▼                  ▼
           Model Performance Comparison
                  │
                  ▼
          Precision / Recall / F1
                  │
                  ▼
                ROC-AUC
```

This makes it possible to determine whether customer segmentation provides useful information for churn prediction.

---

## 📁 Project Structure

```text
live_coding/mlflow/
│
├── data/
│   └── cleaned_data.csv
│
├── pipeline_mlflow.py
│
└── README.md
```

### `data/cleaned_data.csv`

The cleaned customer dataset used for the classification experiments.

### `pipeline_mlflow.py`

Contains the complete ML workflow:

* Data loading
* Train/test split
* Preprocessing
* Model definition
* Pipeline creation
* Model training
* Prediction
* Evaluation
* MLflow experiment tracking
* Model logging

### `README.md`

Project documentation and explanation of the workflow.

---

## 🚀 Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn mlflow
```

---

## ▶️ Run the Project

From the project root:

```bash
python pipeline_mlflow.py
```

This trains the models and creates the corresponding MLflow runs.

---

## 🖥️ Launch the MLflow UI

Start the MLflow tracking interface:

```bash
mlflow ui
```

Then open the address displayed by MLflow in your browser.

The MLflow UI allows you to:

* Compare runs
* Inspect parameters
* Compare metrics
* View logged models
* Explore experiment results

---

## 🔑 Key Concepts Demonstrated

This live coding demonstrates several important machine learning concepts:

* Train/test split
* Stratification
* Feature preprocessing
* `StandardScaler`
* `OneHotEncoder`
* `ColumnTransformer`
* Scikit-learn `Pipeline`
* Logistic Regression
* Random Forest
* Class imbalance
* `class_weight="balanced"`
* `predict()` vs `predict_proba()`
* Precision
* Recall
* F1-score
* ROC-AUC
* MLflow experiments
* MLflow runs
* MLflow parameters
* MLflow metrics
* MLflow model logging
* Model serialization with `cloudpickle`
* Comparing models with and without clustering

---

## 🎓 Live Coding Objective

The main experiment can be summarized as:

> **Can customer segmentation improve the performance of a churn prediction model?**

By comparing the `with_cluster` and `without_cluster` runs in MLflow, we can determine whether the clustering information contributes useful predictive information.

---

## 👨‍💻 Author

**Mohammed Mehdi Saibat**

Machine Learning / Data Engineering Live Coding — YouCode
