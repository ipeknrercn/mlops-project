"""
Simple Drift Detection Module
==============================
Evidently'ye bağımlı olmayan basit drift detection.

PSI (Population Stability Index) kullanır:
- PSI < 0.1  → Değişiklik yok
- 0.1 < PSI < 0.25 → Küçük değişiklik (izle)
- PSI > 0.25 → Büyük değişiklik (ALARM!)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from datetime import datetime
import json
from pathlib import Path


class SimpleDriftDetector:
    """
    Basit ama etkili drift detection.
    Evidently gibi ağır kütüphane yerine PSI hesaplıyor.
    """
    
    def __init__(self, reference_data: pd.DataFrame = None):
        """
        Args:
            reference_data: Eğitimde kullanılan referans data (train set)
        """
        self.reference_data = reference_data
        self.reference_stats = None
        
        if reference_data is not None:
            self.fit_reference(reference_data)
    
    def fit_reference(self, reference_data: pd.DataFrame):
        """Referans datanın istatistiklerini hesapla"""
        self.reference_data = reference_data
        self.reference_stats = {}
        
        for col in reference_data.columns:
            self.reference_stats[col] = {
                'mean': float(reference_data[col].mean()),
                'std': float(reference_data[col].std()),
                'min': float(reference_data[col].min()),
                'max': float(reference_data[col].max()),
                'percentiles': {
                    '25': float(reference_data[col].quantile(0.25)),
                    '50': float(reference_data[col].quantile(0.50)),
                    '75': float(reference_data[col].quantile(0.75))
                }
            }
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        PSI (Population Stability Index) hesapla
        
        Args:
            expected: Referans (train) data
            actual: Production data
            bins: Histogram bin sayısı
            
        Returns:
            PSI değeri
        """
        # Bölme aralıklarını oluştur
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            bins + 1
        )
        
        # Her iki datayı da aynı aralıklarla böl
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Sıfır bölme hatasını önle
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # PSI formülü
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        psi = np.sum(psi_values)
        
        return float(psi)
    
    def detect_drift(self, production_data: pd.DataFrame, threshold: float = 0.25) -> Dict:
        """
        Production datada drift var mı kontrol et
        
        Args:
            production_data: Production'dan gelen yeni data
            threshold: PSI alarm threshold (default: 0.25)
            
        Returns:
            {
                'drift_detected': bool,
                'feature_drifts': {...},
                'overall_psi': float,
                'alert_level': 'OK' | 'WARNING' | 'ALARM'
            }
        """
        if self.reference_data is None:
            raise ValueError("Reference data not fitted! Call fit_reference() first.")
        
        feature_drifts = {}
        psi_scores = []
        
        for col in self.reference_data.columns:
            if col in production_data.columns:
                psi = self.calculate_psi(
                    self.reference_data[col].values,
                    production_data[col].values
                )
                
                feature_drifts[col] = {
                    'psi': psi,
                    'drift': psi > threshold,
                    'severity': self._get_severity(psi)
                }
                psi_scores.append(psi)
        
        # Genel değerlendirme
        overall_psi = np.mean(psi_scores)
        drift_detected = overall_psi > threshold
        
        # Alert level
        if overall_psi < 0.1:
            alert_level = "OK"
        elif overall_psi < 0.25:
            alert_level = "WARNING"
        else:
            alert_level = "ALARM"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'overall_psi': float(overall_psi),
            'alert_level': alert_level,
            'feature_drifts': feature_drifts,
            'n_features_with_drift': sum(1 for f in feature_drifts.values() if f['drift'])
        }
    
    def _get_severity(self, psi: float) -> str:
        """PSI değerine göre severity level döndür"""
        if psi < 0.1:
            return "none"
        elif psi < 0.25:
            return "moderate"
        else:
            return "high"
    
    def save_reference_stats(self, filepath: str):
        """Referans istatistiklerini kaydet"""
        with open(filepath, 'w') as f:
            json.dump(self.reference_stats, f, indent=2)
        print(f"✅ Reference stats saved to {filepath}")
    
    def load_reference_stats(self, filepath: str):
        """Kaydedilmiş referans istatistiklerini yükle"""
        with open(filepath, 'r') as f:
            self.reference_stats = json.load(f)
        print(f"✅ Reference stats loaded from {filepath}")


# ========================================
# PREDICTION DRIFT DETECTION
# ========================================

class PredictionDriftDetector:
    """
    Model prediction'larında drift var mı kontrol et.
    
    Örnek: Eğitimde churn rate %25 idi, production'da %45 olmuş → ALARM!
    """
    
    def __init__(self, reference_predictions: np.ndarray = None):
        """
        Args:
            reference_predictions: Eğitim/test setindeki tahminler
        """
        self.reference_predictions = reference_predictions
        self.reference_mean = None
        self.reference_std = None
        
        if reference_predictions is not None:
            self.fit_reference(reference_predictions)
    
    def fit_reference(self, reference_predictions: np.ndarray):
        """Referans tahminlerin istatistiklerini hesapla"""
        self.reference_predictions = reference_predictions
        self.reference_mean = float(np.mean(reference_predictions))
        self.reference_std = float(np.std(reference_predictions))
    
    def detect_drift(self, production_predictions: np.ndarray, threshold_sigma: float = 2.0) -> Dict:
        """
        Production tahminlerinde drift var mı?
        
        Args:
            production_predictions: Production'dan gelen tahminler
            threshold_sigma: Kaç standart sapma uzaklaşırsa alarm (default: 2.0)
            
        Returns:
            {
                'drift_detected': bool,
                'mean_shift': float,
                'alert_level': str
            }
        """
        prod_mean = float(np.mean(production_predictions))
        prod_std = float(np.std(production_predictions))
        
        # Z-score: referanstan kaç standart sapma uzakta?
        mean_shift = abs(prod_mean - self.reference_mean)
        z_score = mean_shift / (self.reference_std + 1e-10)
        
        drift_detected = z_score > threshold_sigma
        
        # Alert level
        if z_score < 1.0:
            alert_level = "OK"
        elif z_score < threshold_sigma:
            alert_level = "WARNING"
        else:
            alert_level = "ALARM"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'reference_mean': self.reference_mean,
            'production_mean': prod_mean,
            'mean_shift': mean_shift,
            'z_score': float(z_score),
            'alert_level': alert_level
        }


# ========================================
# KULLANIM ÖRNEĞİ
# ========================================

if __name__ == "__main__":
    # Örnek: Dummy data ile test
    np.random.seed(42)
    
    # Referans data (train set)
    reference_df = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(5, 2, 1000),
        'feature_3': np.random.uniform(0, 10, 1000)
    })
    
    # Production data (drift YOK)
    production_no_drift = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 500),
        'feature_2': np.random.normal(5, 2, 500),
        'feature_3': np.random.uniform(0, 10, 500)
    })
    
    # Production data (drift VAR!)
    production_with_drift = pd.DataFrame({
        'feature_1': np.random.normal(2, 1, 500),  # Mean değişmiş!
        'feature_2': np.random.normal(5, 4, 500),  # Variance artmış!
        'feature_3': np.random.uniform(5, 15, 500)  # Range kaymış!
    })
    
    # Test
    detector = SimpleDriftDetector(reference_data=reference_df)
    
    print("🔍 Testing NO DRIFT scenario...")
    result_no_drift = detector.detect_drift(production_no_drift)
    print(f"   Overall PSI: {result_no_drift['overall_psi']:.4f}")
    print(f"   Alert Level: {result_no_drift['alert_level']}")
    
    print("\n🔍 Testing WITH DRIFT scenario...")
    result_with_drift = detector.detect_drift(production_with_drift)
    print(f"   Overall PSI: {result_with_drift['overall_psi']:.4f}")
    print(f"   Alert Level: {result_with_drift['alert_level']}")
    print(f"   Features with drift: {result_with_drift['n_features_with_drift']}/3")
    
    # Feature-level detaylar
    print("\n📊 Feature-level drift scores:")
    for feat, metrics in result_with_drift['feature_drifts'].items():
        print(f"   {feat}: PSI={metrics['psi']:.4f}, Severity={metrics['severity']}")