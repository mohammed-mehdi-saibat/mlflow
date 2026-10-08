import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from mlflow import MlflowClient

# Load Data
df = pd.read_csv("data/cleaned_data.csv")
print(df.shape)

# Check / Create the "cluster" column
if "cluster" not in df.columns:
    cols = ["tenure", "monthly_charges", "total_charges"]
    X_cluster = StandardScaler().fit_transform(df[cols])
    df["cluster"] = KMeans(
        n_clusters=3,
        init="k-means++",
        n_init=10,
        random_state=42
    ).fit_predict(X_cluster)

df["cluster"] = df["cluster"].astype(str)

# Class imbalance

print(df["churn"].value_counts(normalize=True))

# Train Test Split Stratified
X = df.drop(columns=["churn", "customer_id"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Columns
num_cols = [
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
    "total_services_count",
    "avg_monthly_spend_ratio"
]

cat_cols = [c for c in X.columns if c not in num_cols]

print(cat_cols)

# Models 
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
}

# Train + Log Each Run In MLFLOW
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn_prediction")

for use_cluster in [False, True]:
    for name, model in models.items():

        # Categorical columns: remove "cluster" for the "without cluster" runs
        cats = cat_cols if use_cluster else [
            c for c in cat_cols if c != "cluster"
        ]

        # Preprocessing, scaling & one-hot-encoding categories
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats)
        ])

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", clone(model))
        ])

        run_name = f"{name}_{'with' if use_cluster else 'without'}_cluster"

        with mlflow.start_run(run_name=run_name):

            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            mlflow.set_tags({
                "project": "customer_churn_prediction",
                "task": "binary_classification",
                "model_type": name,
                "cluster_feature": str(use_cluster),
                "framework": "scikit-learn",
                "dataset": "cleaned_data.csv"
            })

            mlflow.log_params({
                "model": name,
                "use_cluster": use_cluster,
                "class_weight": "balanced",
                "test_size": 0.2,
                "random_state": 42,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "num_features": len(num_cols),
                "cat_features": len(cats)
            })

            if name == "logistic_regression":
                mlflow.log_params({
                    "max_iter": 1000
                })

            if name == "random_forest":
                mlflow.log_params({
                    "n_estimators": 200,
                    "max_depth": 10
                })

            if use_cluster:
                mlflow.log_params({
                    "n_clusters": 3,
                    "cluster_algorithm": "KMeans",
                    "cluster_features": "tenure,monthly_charges,total_charges",
                    "cluster_scaling": "StandardScaler"
                })

            mlflow.log_metrics({
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            })

            ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred
            )

            plt.title(f"Confusion Matrix - {run_name}")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            plt.close()

            mlflow.log_artifact("confusion_matrix.png", artifact_path="figures")

            mlflow.sklearn.log_model(
                pipe,
                name="model",
                serialization_format="cloudpickle"
            )

        print(f"Done: {run_name}")


# Compare runs
runs = mlflow.search_runs(
    experiment_names=["churn_prediction"],
    order_by=["metrics.roc_auc DESC"]
)

print(
    runs[
        [
            "tags.mlflow.runName",
            "metrics.accuracy",
            "metrics.precision",
            "metrics.recall",
            "metrics.f1",
            "metrics.roc_auc"
        ]
    ].round(3).to_string(index=False)
)

best_name = runs.iloc[0]["tags.mlflow.runName"]
best_run_id = runs.iloc[0]["run_id"]

registered = mlflow.register_model(
    f"runs:/{best_run_id}/model",
    "churn_model"
)

MlflowClient().set_registered_model_alias(
    "churn_model",
    "champion",
    registered.version
)

print(
    f"\nBest model: {best_name} -> "
    f"churn_model version {registered.version}"
)

# Predict the churn of one customer
customer = X_test.iloc[[0]]
model = mlflow.sklearn.load_model(
    "models:/churn_model@champion"
)

proba = model.predict_proba(customer)[0, 1]

print(
    f"Churn probability: {proba:.1%} | "
    f"real value: {y_test.iloc[0]}"
)

# mlflow ui --backend-store-uri sqlite:///mlflow.db
# http://127.0.0.1:5000