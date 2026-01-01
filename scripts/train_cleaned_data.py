"""
Cleaned Kaggle Data Training Script
====================================
Kişi 1'in temizlenmiş datasıyla çalışan versiyon

Farklılıklar:
- customer_id YOK (geo_code zaten var)
- service_combo_id ZATEN VAR (oluşturmaya gerek yok)
- geo_code ZATEN VAR (oluşturmaya gerek yok)
- Sadece Churn Yes/No → 0/1 dönüşümü yapılacak
- Sadece feature crosses oluşturulacak

Kullanım:
    python scripts/train_cleaned_data.py --config config/cleaned_data_config.yaml
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, 
    precision_score, recall_score, classification_report,
    confusion_matrix
)

# Imbalance handling
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


class CleanedDataTrainer:
    """
    Temizlenmiş Kaggle Telco Churn datasıyla model eğitimi
    """
    
    def __init__(self, config_path=None):
        """Initialize with config"""
        project_root = Path(__file__).parent.parent
        
        if config_path is None:
            config_path = project_root / "config" / "cleaned_data_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.experiment_name = self.config.get("experiment_name", "telco-churn-cleaned-data")
        self.random_state = self.config.get("random_state", 42)
        
        # MLflow setup
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()
    
    def load_data(self):
        """
        Temizlenmiş CSV'yi yükle
        """
        print("📥 Loading Cleaned Telco Churn data...")
        
        data_path = self.config["data"]["source"]
        df = pd.read_csv(data_path)
        print(f"   Loaded: {data_path} | shape: {df.shape}")
        
        # Churn to binary (Yes/No → 1/0)
        if df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
        
        # Rename Churn → churn (lowercase)
        df = df.rename(columns={"Churn": "churn"})
        
        # Rename Contract → contract_type
        if "Contract" in df.columns:
            df = df.rename(columns={"Contract": "contract_type"})
        
        # Rename PaymentMethod → payment_method
        if "PaymentMethod" in df.columns:
            df = df.rename(columns={"PaymentMethod": "payment_method"})
        
        # Verify high-cardinality features exist
        required_high_card = ["service_combo_id", "geo_code"]
        missing = [c for c in required_high_card if c not in df.columns]
        if missing:
            raise ValueError(f"Missing high-cardinality features: {missing}")
        
        churn_rate = df['churn'].mean()
        print(f"   Shape: {df.shape}")
        print(f"   Churn rate: {churn_rate:.2%}")
        print(f"   ✅ High-cardinality features found: service_combo_id, geo_code")
        
        return df, churn_rate
    
    def create_feature_crosses(self, df):
        """
        Kişi 2'nin istediği feature crosses
        """
        print("🔀 Creating feature crosses...")
        
        def cross(a, b, sep="__x__"):
            return a.astype(str) + sep + b.astype(str)
        
        # 3 zorunlu cross
        df["cross_contract_payment"] = cross(
            df["contract_type"], df["payment_method"]
        )
        df["cross_service_contract"] = cross(
            df["service_combo_id"], df["contract_type"]
        )
        df["cross_geo_contract"] = cross(
            df["geo_code"], df["contract_type"]
        )
        
        print(f"   Created 3 feature crosses ✅")
        print(f"   - contract_type × payment_method")
        print(f"   - service_combo_id × contract_type")
        print(f"   - geo_code × contract_type")
        
        return df
    
    def prepare_features(self, df):
        """
        X, y ayır ve feature transformers hazırla
        """
        print("📊 Preparing features...")
        
        # Target
        y = df["churn"].astype(int)
        X = df.drop(columns=["churn"])
        
        # Feature groups
        numeric_features = ["tenure", "MonthlyCharges"]
        if "TotalCharges" in X.columns:
            numeric_features.append("TotalCharges")
        
        categorical_features = ["contract_type", "payment_method"]
        # Ekstra kategorikler
        for c in ["gender", "Partner", "Dependents", "PaperlessBilling", "SeniorCitizen"]:
            if c in X.columns:
                categorical_features.append(c)
        
        # Servis kolonları (InternetService, PhoneService, etc.)
        service_cols = [
            "InternetService", "PhoneService", "MultipleLines",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        for c in service_cols:
            if c in X.columns:
                categorical_features.append(c)
        
        cross_features = [
            "cross_contract_payment",
            "cross_service_contract",
            "cross_geo_contract"
        ]
        
        high_card_features = ["service_combo_id", "geo_code"]
        
        print(f"   Numeric: {len(numeric_features)} features")
        print(f"   Categorical: {len(categorical_features)} features")
        print(f"   Crosses: {len(cross_features)} features")
        print(f"   High-card: {len(high_card_features)} features")
        
        return X, y, numeric_features, categorical_features, cross_features, high_card_features
    
    def build_pipeline(self, numeric_features, categorical_features, 
                       cross_features, high_card_features, model_type="baseline"):
        """
        Preprocessing + model pipeline
        """
        print(f"🏗️ Building {model_type} pipeline...")
        
        # Transformers
        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        # High-cardinality hasher (HASHED FEATURE pattern - proje gereksinimi!)
        class HashingTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, cols=None, n_features=2**18):
                self.cols = cols
                self.n_features = n_features
            
            def fit(self, X, y=None):
                # Initialize hasher in fit to avoid serialization issues
                self.hasher_ = FeatureHasher(
                    n_features=self.n_features, input_type="string"
                )
                return self
            
            def transform(self, X):
                tokens = []
                for _, row in X[self.cols].iterrows():
                    row_tokens = [f"{c}={row[c]}" for c in self.cols]
                    tokens.append(row_tokens)
                return self.hasher_.transform(tokens)
        
        hash_transformer = HashingTransformer(
            cols=high_card_features, 
            n_features=2**18  # 262,144 buckets
        )
        
        # ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features + cross_features),
                ("hash", hash_transformer, high_card_features),
            ],
            remainder="drop",
            sparse_threshold=0.3
        )
        
        # Model selection
        if model_type == "baseline":
            model = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",  # REBALANCING pattern
                random_state=self.random_state
            )
        elif model_type == "randomforest":
            # ENSEMBLES pattern - Bagging
            model = RandomForestClassifier(
                n_estimators=400,
                random_state=self.random_state,
                class_weight="balanced_subsample",
                n_jobs=-1
            )
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                # ENSEMBLES pattern - Boosting
                model = XGBClassifier(
                    n_estimators=600,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=self.random_state,
                    eval_metric="logloss",
                    use_label_encoder=False
                )
            except ImportError:
                print("⚠️ XGBoost not available, using RandomForest")
                model = RandomForestClassifier(
                    n_estimators=400,
                    random_state=self.random_state,
                    class_weight="balanced_subsample",
                    n_jobs=-1
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Pipeline with REBALANCING pattern (Upsampling)
        use_oversampling = self.config.get("rebalancing", {}).get("use_oversampling", True)
        
        if use_oversampling:
            sampler = RandomOverSampler(random_state=self.random_state)
            pipeline = ImbPipeline(steps=[
                ("preprocess", preprocessor),
                ("sampler", sampler),  # REBALANCING design pattern
                ("model", model)
            ])
            print(f"   ✅ REBALANCING pattern: RandomOverSampler (Upsampling)")
        else:
            pipeline = ImbPipeline(steps=[
                ("preprocess", preprocessor),
                ("model", model)
            ])
        
        if model_type in ["randomforest", "xgboost"]:
            ensemble_type = "Bagging" if model_type == "randomforest" else "Boosting"
            print(f"   ✅ ENSEMBLES pattern: {ensemble_type}")
        
        return pipeline
    
    def evaluate_model(self, name, pipeline, X_train, y_train, X_test, y_test):
        """
        Model eğit ve değerlendir - PR-AUC primary metric
        """
        print(f"\n🎯 Training {name}...")
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        
        # Metrics - PR-AUC PRIMARY (imbalanced data için en iyi)
        pr_auc = average_precision_score(y_test, proba)
        roc_auc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        
        print(f"\n{'='*60}")
        print(f"📊 {name} - RESULTS")
        print(f"{'='*60}")
        print(f"PRIMARY METRIC (imbalanced data):")
        print(f"  PR-AUC:    {pr_auc:.4f} ⭐")
        print(f"\nSECONDARY METRICS:")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"{'='*60}\n")
        
        print(classification_report(y_test, pred, digits=3))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, pred)
        
        return {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "proba": proba,
            "pred": pred,
            "confusion_matrix": cm
        }
    
    def plot_confusion_matrix(self, cm, model_name):
        """Confusion matrix görselleştirme"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        # Save
        os.makedirs("plots", exist_ok=True)
        plot_path = f"plots/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def log_to_mlflow(self, model_name, pipeline, metrics, params, plot_path=None):
        """MLflow'a log et"""
        print(f"📝 Logging {model_name} to MLflow...")
        
        with mlflow.start_run(run_name=model_name):
            # Params
            for k, v in params.items():
                mlflow.log_param(k, v)
            
            # Metrics - PR-AUC PRIMARY!
            mlflow.log_metric("pr_auc", float(metrics["pr_auc"]))
            mlflow.log_metric("roc_auc", float(metrics["roc_auc"]))
            mlflow.log_metric("f1", float(metrics["f1"]))
            mlflow.log_metric("precision", float(metrics["precision"]))
            mlflow.log_metric("recall", float(metrics["recall"]))
            
            # Model
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            
            # Plot
            if plot_path and os.path.exists(plot_path):
                mlflow.log_artifact(plot_path)
            
            run_id = mlflow.active_run().info.run_id
            print(f"   ✅ Logged run: {run_id}")
            
            # Production readiness check
            thresholds = self.config.get("thresholds", {})
            min_pr_auc = thresholds.get("production_min_pr_auc", 0.65)
            min_roc_auc = thresholds.get("production_min_roc_auc", 0.70)
            min_recall = thresholds.get("production_min_recall", 0.60)
            
            checks = {
                "pr_auc_check": metrics["pr_auc"] >= min_pr_auc,
                "roc_auc_check": metrics["roc_auc"] >= min_roc_auc,
                "recall_check": metrics["recall"] >= min_recall
            }
            
            ready_for_production = all(checks.values())
            
            if ready_for_production:
                print(f"   ✅ Model READY for production!")
                stage = "Production"
            else:
                print(f"   ⚠️ Model NOT ready for production")
                failed = [k for k, v in checks.items() if not v]
                print(f"      Failed checks: {failed}")
                stage = "Staging"
            
            # Register model
            model_name_registry = self.config.get("registry", {}).get(
                "model_name", "telco_churn_classifier_cleaned"
            )
            
            # Register
            result = mlflow.register_model(
                f"runs:/{run_id}/model",
                model_name_registry
            )
            
            # Transition to stage
            self.client.transition_model_version_stage(
                name=model_name_registry,
                version=result.version,
                stage=stage
            )
            
            print(f"   📦 Registered as {model_name_registry} v{result.version} ({stage})")
            
            return run_id, result.version, stage
    
    def run(self):
        """
        End-to-end training pipeline
        """
        print("\n" + "="*80)
        print("🚀 CLEANED DATA TRAINING PIPELINE")
        print("="*80 + "\n")
        
        # 1. Load data
        df, churn_rate = self.load_data()
        
        # 2. Feature crosses (service_combo_id ve geo_code zaten var!)
        df = self.create_feature_crosses(df)
        
        # 3. Prepare features
        X, y, numeric_features, categorical_features, cross_features, high_card_features = self.prepare_features(df)
        
        # 4. Train/test split
        test_size = self.config["data"].get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"\n📊 Dataset Split:")
        print(f"   Train: {X_train.shape[0]} samples | churn rate: {y_train.mean():.2%}")
        print(f"   Test:  {X_test.shape[0]} samples | churn rate: {y_test.mean():.2%}")
        
        # 5. Baseline model
        baseline_pipeline = self.build_pipeline(
            numeric_features, categorical_features, 
            cross_features, high_card_features,
            model_type="baseline"
        )
        
        baseline_metrics = self.evaluate_model(
            "Baseline (LogisticRegression)",
            baseline_pipeline,
            X_train, y_train,
            X_test, y_test
        )
        
        baseline_plot = self.plot_confusion_matrix(
            baseline_metrics["confusion_matrix"],
            "Baseline_LogReg"
        )
        
        baseline_run, baseline_version, baseline_stage = self.log_to_mlflow(
            "baseline_logreg_cleaned",
            baseline_pipeline,
            baseline_metrics,
            params={
                "model_type": "logistic_regression",
                "oversampling": True,
                "hash_dim": 2**18,
                "dataset": "kaggle_telco_churn_cleaned",
                "high_card_features": "service_combo_id,geo_code",
                "feature_crosses": "3_crosses"
            },
            plot_path=baseline_plot
        )
        
        # 6. Main model (XGBoost or RandomForest)
        main_model_type = self.config.get("model", {}).get("main", {}).get("type", "xgboost")
        
        main_pipeline = self.build_pipeline(
            numeric_features, categorical_features,
            cross_features, high_card_features,
            model_type=main_model_type
        )
        
        main_metrics = self.evaluate_model(
            f"Main ({main_model_type})",
            main_pipeline,
            X_train, y_train,
            X_test, y_test
        )
        
        main_plot = self.plot_confusion_matrix(
            main_metrics["confusion_matrix"],
            f"Main_{main_model_type}"
        )
        
        main_run, main_version, main_stage = self.log_to_mlflow(
            f"main_{main_model_type}_cleaned",
            main_pipeline,
            main_metrics,
            params={
                "model_type": main_model_type,
                "oversampling": True,
                "hash_dim": 2**18,
                "dataset": "kaggle_telco_churn_cleaned",
                "high_card_features": "service_combo_id,geo_code",
                "feature_crosses": "3_crosses"
            },
            plot_path=main_plot
        )
        
        # 7. Best model
        if main_metrics["pr_auc"] >= baseline_metrics["pr_auc"]:
            best_name = f"Main ({main_model_type})"
            best_pr = main_metrics["pr_auc"]
            best_version = main_version
            best_stage = main_stage
        else:
            best_name = "Baseline (LogReg)"
            best_pr = baseline_metrics["pr_auc"]
            best_version = baseline_version
            best_stage = baseline_stage
        
        print("\n" + "="*80)
        print("🏆 FINAL RESULTS")
        print("="*80)
        print(f"Best Model: {best_name}")
        print(f"PR-AUC: {best_pr:.4f}")
        print(f"Version: {best_version}")
        print(f"Stage: {best_stage}")
        print("="*80 + "\n")
        
        print("\n✅ MLOps Design Patterns Implemented:")
        print("   1. HASHED FEATURE: High-cardinality → 2^18 buckets")
        print("   2. FEATURE CROSS: 3 crosses (contract×payment, service×contract, geo×contract)")
        print("   3. REBALANCING: RandomOverSampler (Upsampling)")
        print("   4. ENSEMBLES: XGBoost (Boosting) / RandomForest (Bagging)")
        print("   5. Primary Metric: PR-AUC (imbalanced data optimized)")
        
        return {
            "baseline": baseline_metrics,
            "main": main_metrics,
            "best": {
                "name": best_name,
                "pr_auc": best_pr,
                "version": best_version,
                "stage": best_stage
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/cleaned_data_config.yaml",
        help="Config file path"
    )
    args = parser.parse_args()
    
    trainer = CleanedDataTrainer(config_path=args.config)
    results = trainer.run()
    
    print("\n✅ Training completed!")
    print(f"   Best PR-AUC: {results['best']['pr_auc']:.4f}")
    print(f"   Model version: {results['best']['version']}")
    print(f"   Stage: {results['best']['stage']}")