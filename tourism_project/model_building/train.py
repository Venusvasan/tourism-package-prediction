import os
import joblib
import pandas as pd
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

# Set the tracking URL for MLflow
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
# Set the name for the experiment
mlflow.set_experiment("Wellness_Tourism_Package_Prediction")


X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")
y_train = pd.read_csv("ytrain.csv").squeeze()
y_test = pd.read_csv("ytest.csv").squeeze()
#  Identify numerical and categorical columns
num_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()

#Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), num_features),
    (OneHotEncoder(handle_unknown="ignore"), cat_features)
)

# Define XGBoost Classifier
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Define hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
  
    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

# Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)
 # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45
    y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })
# Save next to app.py so the Streamlit app can load it directly, and log
# it as an MLflow artifact for traceability
os.makedirs("tourism_project/deployment", exist_ok=True)
model_path = "tourism_project/deployment/best_model.joblib"
joblib.dump(best_model, model_path)
mlflow.log_artifact(model_path, artifact_path="model")
print(f"Model saved to {model_path}")
