"""
Unit Tests for Feature Engineering
These tests are FAST (< 1s) with NO external dependencies
"""
import pytest
import pandas as pd
import sys
sys.path.append('..')

def create_feature_crosses_standalone(df):
    """Standalone version for testing"""
    df["cross_contract_payment"] = (
        df["contract_type"].astype(str) + "__x__" + 
        df["payment_method"].astype(str)
    )
    df["cross_service_contract"] = (
        df["service_combo_id"].astype(str) + "__x__" + 
        df["contract_type"].astype(str)
    )
    df["cross_geo_contract"] = (
        df["geo_code"].astype(str) + "__x__" + 
        df["contract_type"].astype(str)
    )
    return df


class TestFeatureCrosses:
    """Test suite for feature cross generation"""
    
    def test_feature_cross_creation(self):
        """Test: Feature crosses are created with correct separator"""
        # Arrange
        df = pd.DataFrame({
            'contract_type': ['Month-to-month'],
            'payment_method': ['Electronic check'],
            'service_combo_id': ['ServiceA'],
            'geo_code': ['G01']
        })
        
        # Act
        result = create_feature_crosses_standalone(df)
        
        # Assert
        assert 'cross_contract_payment' in result.columns
        assert result['cross_contract_payment'][0] == 'Month-to-month__x__Electronic check'
        
    def test_feature_cross_count(self):
        """Test: Exactly 3 feature crosses created"""
        df = pd.DataFrame({
            'contract_type': ['One year'],
            'payment_method': ['Bank transfer'],
            'service_combo_id': ['ServiceB'],
            'geo_code': ['G02']
        })
        
        result = create_feature_crosses_standalone(df)
        
        # Should have 3 new columns
        cross_cols = [col for col in result.columns if col.startswith('cross_')]
        assert len(cross_cols) == 3
    
    def test_feature_cross_deterministic(self):
        """Test: Same input produces same output (deterministic)"""
        df = pd.DataFrame({
            'contract_type': ['Two year'],
            'payment_method': ['Credit card'],
            'service_combo_id': ['ServiceC'],
            'geo_code': ['G03']
        })
        
        result1 = create_feature_crosses_standalone(df.copy())
        result2 = create_feature_crosses_standalone(df.copy())
        
        assert result1['cross_contract_payment'][0] == result2['cross_contract_payment'][0]


class TestDataValidation:
    """Test suite for data validation logic"""
    
    def test_missing_values_handling(self):
        """Test: Missing values are handled correctly"""
        df = pd.DataFrame({
            'contract_type': ['Month-to-month', None],
            'payment_method': ['Electronic check', 'Cash'],
            'service_combo_id': ['ServiceA', 'ServiceB'],
            'geo_code': ['G01', 'G02']
        })
        
        result = create_feature_crosses_standalone(df)
        
        # Should not crash - None converted to string "None"
        assert len(result) == 2
        assert 'None' in result['cross_contract_payment'][1]


# WHY THESE TESTS ARE FAST:
# 1. No file I/O (no CSV loading)
# 2. No network calls (no API requests)
# 3. No database connections
# 4. Pure computation on small DataFrames
# 5. Runs in < 1 second total