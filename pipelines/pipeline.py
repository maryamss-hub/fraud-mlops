"""Kubeflow Pipeline (KFP v2) for fraud detection MLOps."""

import os

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Model, Output


BASE_IMAGE = "python:3.10-slim"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "numpy"],
)
def data_ingestion(
    transactions_csv: str,
    identity_csv: str,
    merged_dataset: Output[Dataset],
) -> None:
    """Load and merge raw CSV inputs."""
    import os

    import pandas as pd

    merged_path = merged_dataset.path
    os.makedirs(os.path.dirname(merged_path) or ".", exist_ok=True)
    df_t = pd.read_csv(transactions_csv)
    df_i = pd.read_csv(identity_csv)
    df = df_t.merge(df_i, on="TransactionID", how="left")
    df.to_csv(merged_path, index=False)
    merged_dataset.metadata["format"] = "csv"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas"],
)
def data_validation(merged_dataset: Input[Dataset], report: Output[Dataset]) -> None:
    """Run basic schema validation and write a report."""
    import os

    import pandas as pd

    df = pd.read_csv(merged_dataset.path)
    failures = []
    for col in ("TransactionID", "TransactionDT"):
        if col not in df.columns:
            failures.append(f"Missing required column: {col}")
    report_path = report.path
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        if failures:
            f.write("FAILED\n")
            for msg in failures:
                f.write(msg + "\n")
        else:
            f.write("SUCCESS\n")
    report.metadata["type"] = "text"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "numpy"],
)
def preprocessing(merged_dataset: Input[Dataset], cleaned_dataset: Output[Dataset]) -> None:
    """Handle missing values and write cleaned dataset."""
    import os

    import numpy as np
    import pandas as pd

    df = pd.read_csv(merged_dataset.path)
    missing_frac = df.isna().mean()
    drop_cols = missing_frac[missing_frac > 0.50].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)

    cleaned_path = cleaned_dataset.path
    os.makedirs(os.path.dirname(cleaned_path) or ".", exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    cleaned_dataset.metadata["format"] = "csv"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "numpy", "category_encoders"],
)
def feature_engineering(cleaned_dataset: Input[Dataset], fe_dataset: Output[Dataset]) -> None:
    """Encode categoricals and add basic engineered features."""
    import os

    import numpy as np
    import pandas as pd
    from category_encoders.target_encoder import TargetEncoder

    df = pd.read_csv(cleaned_dataset.path)
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"].astype(float))

    target_col = "isFraud"
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    high = [c for c in cat_cols if c in {"card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceInfo"}]
    low = [c for c in cat_cols if c not in high]
    if target_col in df.columns and high:
        enc = TargetEncoder(cols=high, smoothing=10.0)
        df[high] = enc.fit_transform(df[high], df[target_col])
    for c in low:
        df[c] = pd.factorize(df[c].astype(str), sort=True)[0].astype(np.int32)

    fe_path = fe_dataset.path
    os.makedirs(os.path.dirname(fe_path) or ".", exist_ok=True)
    df.to_csv(fe_path, index=False)
    fe_dataset.metadata["format"] = "csv"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "numpy", "xgboost", "scikit-learn", "joblib"],
)
def model_training(fe_dataset: Input[Dataset], model_artifact: Output[Model]) -> None:
    """Train an XGBoost model and save it."""
    import os

    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    df = pd.read_csv(fe_dataset.path)
    if "isFraud" not in df.columns:
        raise ValueError("Missing target column 'isFraud'")
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"].astype(int)
    X_tr, _X_te, y_tr, _y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=0,
    )
    model.fit(X_tr.values, y_tr.values)
    model_path = model_artifact.path
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)
    model_artifact.metadata["framework"] = "xgboost"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"],
)
def model_evaluation(fe_dataset: Input[Dataset], model_artifact: Input[Model]) -> float:
    """Evaluate the model and return recall (used for conditional deploy)."""
    import joblib
    import pandas as pd
    from sklearn.metrics import recall_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(fe_dataset.path)
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"].astype(int)
    _X_tr, X_te, _y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = joblib.load(model_artifact.path)
    prob = model.predict_proba(X_te.values)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return float(recall_score(y_te.values, pred, zero_division=0))


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=[],
)
def conditional_deploy(model_artifact: Input[Model], deployed_model: Output[Model]) -> None:
    """Deploy by copying a model artifact to a deployed path."""
    import os
    import shutil

    deployed_path = deployed_model.path
    os.makedirs(os.path.dirname(deployed_path) or ".", exist_ok=True)
    shutil.copyfile(model_artifact.path, deployed_path)
    deployed_model.metadata["deployed"] = "true"


@dsl.pipeline(name="fraud-detection-mlops-pipeline")
def fraud_pipeline(transactions_csv: str, identity_csv: str):
    """End-to-end fraud MLOps pipeline."""
    ingest_task = data_ingestion(transactions_csv=transactions_csv, identity_csv=identity_csv).set_retry(
        num_retries=3
    )
    _ = data_validation(merged_dataset=ingest_task.outputs["merged_dataset"]).set_retry(num_retries=3)
    preprocess_task = preprocessing(merged_dataset=ingest_task.outputs["merged_dataset"]).set_retry(num_retries=3)
    fe_task = feature_engineering(cleaned_dataset=preprocess_task.outputs["cleaned_dataset"]).set_retry(
        num_retries=3
    )
    train_task = model_training(fe_dataset=fe_task.outputs["fe_dataset"]).set_retry(num_retries=3)
    eval_task = model_evaluation(fe_dataset=fe_task.outputs["fe_dataset"], model_artifact=train_task.outputs["model_artifact"]).set_retry(
        num_retries=3
    )

    with dsl.Condition(eval_task.output > 0.85, name="deploy_if_good_recall"):
        _ = conditional_deploy(model_artifact=train_task.outputs["model_artifact"]).set_retry(num_retries=3)


def compile_pipeline(output_path: str = "pipelines/fraud_pipeline.yaml") -> str:
    """Compile the pipeline to a YAML spec."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    compiler.Compiler().compile(pipeline_func=fraud_pipeline, package_path=output_path)
    return output_path


if __name__ == "__main__":
    compile_pipeline()

