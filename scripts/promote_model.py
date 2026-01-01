"""
Model Promotion Script
======================
Staging'deki modeli Production'a taşır.

Kullanım:
    python promote_model.py --version 3
    python promote_model.py --auto  # En son staging modelini al
"""

import mlflow
from mlflow.tracking import MlflowClient
import argparse
from datetime import datetime


class ModelPromoter:
    """Model promotion yönetimi"""
    
    def __init__(self, model_name: str = "telco_churn_classifier"):
        self.model_name = model_name
        self.client = MlflowClient()
    
    def get_staging_models(self):
        """Staging'deki tüm modelleri getir"""
        staging_models = self.client.get_latest_versions(
            self.model_name, 
            stages=["Staging"]
        )
        return staging_models
    
    def get_production_models(self):
        """Production'daki tüm modelleri getir"""
        prod_models = self.client.get_latest_versions(
            self.model_name, 
            stages=["Production"]
        )
        return prod_models
    
    def get_model_metrics(self, version: str):
        """Model version'ına ait metrikleri getir"""
        model_version = self.client.get_model_version(self.model_name, version)
        run_id = model_version.run_id
        
        run = self.client.get_run(run_id)
        metrics = run.data.metrics
        
        # Try both old (test_*) and new (*) metric naming conventions
        roc_auc = (
            metrics.get("test_roc_auc") or 
            metrics.get("roc_auc") or 
            metrics.get("val_roc_auc")
        )
        pr_auc = (
            metrics.get("test_pr_auc") or 
            metrics.get("pr_auc") or 
            metrics.get("val_pr_auc")
        )
        
        # Return normalized dict
        return {
            "test_roc_auc": roc_auc,
            "test_pr_auc": pr_auc
        }
    
    def compare_models(self, new_version: str, current_prod_version: str = None):
        """Yeni model ile production modelini karşılaştır"""
        new_metrics = self.get_model_metrics(new_version)
        
        comparison = {
            "new_model_version": new_version,
            "new_model_metrics": new_metrics
        }
        
        if current_prod_version:
            prod_metrics = self.get_model_metrics(current_prod_version)
            comparison["current_prod_version"] = current_prod_version
            comparison["current_prod_metrics"] = prod_metrics
            
            # Karşılaştırma - both naming conventions
            new_roc = new_metrics.get("test_roc_auc") or new_metrics.get("roc_auc", 0)
            new_pr = new_metrics.get("test_pr_auc") or new_metrics.get("pr_auc", 0)
            new_recall = new_metrics.get("test_recall") or new_metrics.get("recall", 0)
            
            prod_roc = prod_metrics.get("test_roc_auc") or prod_metrics.get("roc_auc", 0)
            prod_pr = prod_metrics.get("test_pr_auc") or prod_metrics.get("pr_auc", 0)
            prod_recall = prod_metrics.get("test_recall") or prod_metrics.get("recall", 0)
            
            comparison["improvements"] = {
                "roc_auc_delta": new_roc - prod_roc,
                "pr_auc_delta": new_pr - prod_pr,
                "recall_delta": new_recall - prod_recall
            }
            
            # Karar: yeni model daha mı iyi?
            comparison["is_improvement"] = (
                comparison["improvements"]["roc_auc_delta"] >= 0 and
                comparison["improvements"]["pr_auc_delta"] >= 0 and
                comparison["improvements"]["recall_delta"] >= -0.02  # Recall'da max %2 düşüşe izin
            )
        else:
            # İlk production modeli
            comparison["is_improvement"] = True
        
        return comparison
    
    def promote_to_production(self, version: str, archive_old: bool = True):
        """
        Model'i production'a taşı
        
        Args:
            version: Model version numarası
            archive_old: Eski production modelini arşivle
        """
        print(f"\n🚀 Promoting model version {version} to PRODUCTION...")
        
        # Önce staging'de mi kontrol et
        model_version = self.client.get_model_version(self.model_name, version)
        if model_version.current_stage != "Staging":
            print(f"❌ Error: Model version {version} is not in Staging stage!")
            print(f"   Current stage: {model_version.current_stage}")
            return False
        
        # Mevcut production modeli var mı?
        prod_models = self.get_production_models()
        
        if prod_models:
            current_prod_version = prod_models[0].version
            
            # Karşılaştırma yap
            comparison = self.compare_models(version, current_prod_version)
            
            print("\n📊 Model Comparison:")
            print(f"   Current Production: v{current_prod_version}")
            print(f"   New Candidate: v{version}")
            print(f"\n   Metrics Delta:")
            for metric, delta in comparison["improvements"].items():
                emoji = "📈" if delta >= 0 else "📉"
                print(f"   {emoji} {metric}: {delta:+.4f}")
            
            if not comparison["is_improvement"]:
                print("\n⚠️ Warning: New model is NOT better than current production!")
                confirm = input("   Continue anyway? (yes/no): ")
                if confirm.lower() != "yes":
                    print("   Promotion cancelled.")
                    return False
        
        # Production'a taşı
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=archive_old
        )
        
        # Description güncelle
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.client.update_model_version(
            name=self.model_name,
            version=version,
            description=f"Promoted to Production on {timestamp}"
        )
        
        print(f"\n✅ Model version {version} is now in PRODUCTION!")
        
        if archive_old and prod_models:
            print(f"   Previous version {current_prod_version} archived.")
        
        return True
    
    def rollback_to_version(self, version: str):
        """Belirli bir versiyona rollback yap"""
        print(f"\n🔄 Rolling back to version {version}...")
        
        # Version'ı production'a al
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.client.update_model_version(
            name=self.model_name,
            version=version,
            description=f"Rollback to this version on {timestamp}"
        )
        
        print(f"✅ Rolled back to version {version}")
        return True
    
    def list_all_versions(self):
        """Tüm model versiyonlarını listele"""
        print(f"\n📋 All versions of '{self.model_name}':")
        print("="*80)
        
        for stage in ["Production", "Staging", "Archived", "None"]:
            versions = self.client.get_latest_versions(self.model_name, stages=[stage])
            if versions:
                print(f"\n{stage.upper()}:")
                for v in versions:
                    metrics = self.get_model_metrics(v.version)
                    
                    # Try both naming conventions (test_ prefix and without)
                    roc_auc_val = metrics.get("test_roc_auc") or metrics.get("roc_auc")
                    pr_auc_val = metrics.get("test_pr_auc") or metrics.get("pr_auc")
                    
                    # Safe formatting
                    roc_str = f"{roc_auc_val:.4f}" if isinstance(roc_auc_val, (int, float)) else "N/A"
                    pr_str = f"{pr_auc_val:.4f}" if isinstance(pr_auc_val, (int, float)) else "N/A"
                    
                    print(f"  Version {v.version}: ROC-AUC={roc_str}, PR-AUC={pr_str}")


def main():
    parser = argparse.ArgumentParser(description="Promote model to production")
    parser.add_argument("--version", type=str, help="Model version to promote")
    parser.add_argument("--auto", action="store_true", help="Auto-promote latest staging model")
    parser.add_argument("--rollback", type=str, help="Rollback to specific version")
    parser.add_argument("--list", action="store_true", help="List all model versions")
    parser.add_argument("--model-name", type=str, default="telco_churn_classifier_cleaned", help="Model name")
    
    args = parser.parse_args()
    
    promoter = ModelPromoter(model_name=args.model_name)
    
    if args.list:
        promoter.list_all_versions()
    
    elif args.rollback:
        promoter.rollback_to_version(args.rollback)
    
    elif args.auto:
        # En son staging modelini al
        staging_models = promoter.get_staging_models()
        if not staging_models:
            print("❌ No models in Staging!")
            return
        
        latest_staging = staging_models[0]
        print(f"📌 Latest staging model: version {latest_staging.version}")
        promoter.promote_to_production(latest_staging.version)
    
    elif args.version:
        promoter.promote_to_production(args.version)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()