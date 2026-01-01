"""
Telco Churn MLOps - Improved Training Script
============================================
Bu script:
1. Config dosyasından parametreleri okur
2. Model signature ile MLflow'a kaydeder
3. Validation set kullanır
4. Model Registry'e (staging) kaydeder
5. Promotion için metrikleri değerlendirir
6. Görselleştirmeleri loglar
"""

# CI/CD ortamlarında matplotlib backend hatalarını önlemek için
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.model_selection import train_test_split
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# YAML import kontrolü
try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML not installed! Please run: pip install PyYAML"
    )


class ChurnTrainer:
    """Churn model training ve MLflow entegrasyonu"""
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: YAML config dosyası yolu
        """
        self.config = self._load_config(config_path)
        self.experiment_name = self.config.get("experiment_name", "telco-churn-mlops")
        mlflow.set_experiment(self.experiment_name)
        
    def _load_config(self, config_path: str) -> dict:
        """Config dosyasını yükle veya default değerleri kullan"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default config (config dosyası yoksa)
        return {
            "experiment_name": "telco-churn-mlops",
            "data": {
                "n_samples": 4000,
                "n_features": 20,
                "n_informative": 10,
                "n_redundant": 4,
                "test_size": 0.2,
                "val_size": 0.15,
                "imbalance_ratio": [0.75, 0.25],
                "random_state": 42
            },
            "model": {
                "type": "LogisticRegression",
                "max_iter": 300,
                "class_weight": None  # None veya "balanced"
            },
            "thresholds": {
                "production_min_roc_auc": 0.70,
                "production_min_pr_auc": 0.65,
                "production_min_recall": 0.50
            }
        }
    
    def generate_data(self):
        """Dummy dataset oluştur (imbalanced)"""
        data_config = self.config["data"]
        
        X, y = make_classification(
            n_samples=data_config["n_samples"],
            n_features=data_config["n_features"],
            n_informative=data_config["n_informative"],
            n_redundant=data_config["n_redundant"],
            weights=data_config["imbalance_ratio"],
            random_state=data_config["random_state"],
        )
        
        # Train-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=data_config["test_size"], 
            random_state=data_config["random_state"], 
            stratify=y
        )
        
        # Train-validation split
        val_ratio = data_config["val_size"] / (1 - data_config["test_size"])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=data_config["random_state"],
            stratify=y_temp
        )
        
        print(f"📊 Dataset split:")
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Val:   {X_val.shape[0]} samples")
        print(f"   Test:  {X_test.shape[0]} samples")
        print(f"   Churn ratio: {y.mean():.2%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, y_train):
        """Model eğit"""
        model_config = self.config["model"]
        
        if model_config["type"] == "LogisticRegression":
            model = LogisticRegression(
                max_iter=model_config["max_iter"],
                class_weight=model_config.get("class_weight"),
                random_state=self.config["data"]["random_state"]
            )
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
        
        model.fit(X_train, y_train)
        return model
    
    def evaluate(self, model, X, y, dataset_name="test"):
        """Model değerlendirme - comprehensive metrics"""
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        
        metrics = {
            f"{dataset_name}_roc_auc": float(roc_auc_score(y, proba)),
            f"{dataset_name}_pr_auc": float(average_precision_score(y, proba)),
            f"{dataset_name}_precision": float(precision_score(y, pred, zero_division=0)),
            f"{dataset_name}_recall": float(recall_score(y, pred, zero_division=0)),
            f"{dataset_name}_f1": float(2 * precision_score(y, pred, zero_division=0) * recall_score(y, pred, zero_division=0) / 
                                       (precision_score(y, pred, zero_division=0) + recall_score(y, pred, zero_division=0) + 1e-10))
        }
        
        return metrics, proba, pred
    
    def plot_metrics(self, y_true, y_proba, y_pred):
        """Confusion matrix ve ROC curve çiz"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm[i, j]), 
                           ha='center', va='center', color='red', fontsize=14)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        axes[1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_true, y_proba):.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def check_production_readiness(self, metrics: dict) -> dict:
        """Production'a çıkmaya hazır mı kontrol et"""
        thresholds = self.config["thresholds"]
        
        checks = {
            "roc_auc_check": metrics["test_roc_auc"] >= thresholds["production_min_roc_auc"],
            "pr_auc_check": metrics["test_pr_auc"] >= thresholds["production_min_pr_auc"],
            "recall_check": metrics["test_recall"] >= thresholds["production_min_recall"]
        }
        
        checks["ready_for_production"] = all(checks.values())
        
        return checks
    
    def run(self, register_model: bool = True):
        """Ana training pipeline"""
        
        run_name = f"churn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            print(f"\n🚀 Starting MLflow run: {run_name}")
            print(f"   Run ID: {run.info.run_id}")
            
            # 1. Data Generation
            X_train, X_val, X_test, y_train, y_val, y_test = self.generate_data()
            
            # 2. Log Parameters
            mlflow.log_params({
                "model_type": self.config["model"]["type"],
                "n_features": X_train.shape[1],
                "n_train_samples": X_train.shape[0],
                "n_val_samples": X_val.shape[0],
                "n_test_samples": X_test.shape[0],
                "class_weight": str(self.config["model"].get("class_weight")),
                "max_iter": self.config["model"]["max_iter"]
            })
            
            # 3. Train Model
            print("\n🎯 Training model...")
            model = self.train_model(X_train, y_train)
            
            # 4. Evaluate on Validation
            print("📊 Evaluating on validation set...")
            val_metrics, val_proba, val_pred = self.evaluate(model, X_val, y_val, "val")
            
            # 5. Evaluate on Test
            print("📊 Evaluating on test set...")
            test_metrics, test_proba, test_pred = self.evaluate(model, X_test, y_test, "test")
            
            # Combine metrics
            all_metrics = {**val_metrics, **test_metrics}
            mlflow.log_metrics(all_metrics)
            
            # 6. Plot and log visualizations
            print("📈 Creating visualizations...")
            fig = self.plot_metrics(y_test, test_proba, test_pred)
            mlflow.log_figure(fig, "metrics_visualization.png")
            plt.close(fig)
            
            # 7. Check production readiness
            production_check = self.check_production_readiness(all_metrics)
            mlflow.log_dict(production_check, "production_readiness.json")
            
            # 8. Model signature (input/output şeması)
            signature = infer_signature(X_train, model.predict_proba(X_train))
            
            # 9. Log model with signature
            print("Logging model to MLflow...")
            mlflow.sklearn.log_model(
                model, 
                artifact_path="model",
                signature=signature,
                input_example=X_train[:5]  # İlk 5 satır örnek
            )
            
            # 10. Register to Model Registry
            if register_model:
                print("Registering model to Model Registry...")
                model_uri = f"runs:/{run.info.run_id}/model"
                model_name = self.config.get("registry", {}).get("model_name", "telco_churn_classifier")
                
                mv = mlflow.register_model(model_uri, model_name)
                
                # Production'a hazırsa staging'e al
                client = mlflow.tracking.MlflowClient()
                if production_check["ready_for_production"]:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=mv.version,
                        stage="Staging",
                        archive_existing_versions=False
                    )
                    print(f"Model version {mv.version} moved to STAGING")
                else:
                    print(f"Model version {mv.version} registered but NOT ready for production")
                    print(f"   Failed checks: {[k for k, v in production_check.items() if not v and k != 'ready_for_production']}")
            
            # 11. Print Summary
            print("\n" + "="*60)
            print("📊 TRAINING SUMMARY")
            print("="*60)
            print(f"Test ROC-AUC:  {all_metrics['test_roc_auc']:.4f}")
            print(f"Test PR-AUC:   {all_metrics['test_pr_auc']:.4f}")
            print(f"Test Precision: {all_metrics['test_precision']:.4f}")
            print(f"Test Recall:    {all_metrics['test_recall']:.4f}")
            print(f"Test F1:        {all_metrics['test_f1']:.4f}")
            print(f"\nProduction Ready: {'✅ YES' if production_check['ready_for_production'] else '❌ NO'}")
            print("="*60)
            
            return run.info.run_id, all_metrics, production_check


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Churn Model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--no-register", action="store_true", help="Don't register model to registry")
    
    args = parser.parse_args()
    
    trainer = ChurnTrainer(config_path=args.config)
    run_id, metrics, checks = trainer.run(register_model=not args.no_register)
    
    print(f"\n✅ Training completed! Run ID: {run_id}")