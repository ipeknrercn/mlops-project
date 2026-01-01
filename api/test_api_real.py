"""
API Test Script - Real Telco Data
==================================
Tests the prediction API with real telco customer features

Usage:
    python api/test_api_real.py
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)


def test_health_check():
    """Test 1: Health check endpoint"""
    print_section("TEST 1: HEALTH CHECK")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data.get('status')}")
            print(f"✅ Model Loaded: {data.get('model_loaded')}")
            print(f"✅ Model Version: {data.get('model_version')}")
            print(f"✅ Timestamp: {data.get('timestamp')}")
            return True
        else:
            print(f"❌ FAILED with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_model_info():
    """Test 2: Model info endpoint"""
    print_section("TEST 2: MODEL INFO")
    
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model Name: {data.get('model_name')}")
            print(f"✅ Model Version: {data.get('model_version')}")
            print(f"✅ Model Stage: {data.get('model_stage')}")
            print(f"✅ Loaded At: {data.get('loaded_at')}")
            
            metrics = data.get('metrics', {})
            if metrics:
                print(f"✅ Metrics Available: {len(metrics)} metrics")
            return True
        else:
            print(f"❌ FAILED with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_single_prediction():
    """Test 3: Single customer prediction"""
    print_section("TEST 3: SINGLE PREDICTION")
    
    # Real telco customer data (example)
    # These features match the cleaned Kaggle dataset
    customer = {
        "customer_id": "TEST-CUST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "contract_type": "Month-to-month",
        "PaperlessBilling": "Yes",
        "payment_method": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 1078.20,
        "service_combo_id": "Fiber optic_Yes_Yes_Yes",
        "geo_code": "G23"
    }
    
    try:
        print("📤 Sending request...")
        print(f"   Customer ID: {customer['customer_id']}")
        print(f"   Tenure: {customer['tenure']} months")
        print(f"   Monthly Charges: ${customer['MonthlyCharges']}")
        print(f"   Contract: {customer['contract_type']}")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=customer,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\n📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction Successful!")
            print(f"   Churn Probability: {data.get('churn_probability', 0):.4f}")
            print(f"   Churn Prediction: {data.get('churn_prediction')}")
            print(f"   Risk Level: {data.get('risk_level')}")
            print(f"   Confidence: {data.get('confidence', 0):.2%}")
            return True
        else:
            print(f"❌ FAILED with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_batch_predictions():
    """Test 4: Multiple customers"""
    print_section("TEST 4: BATCH PREDICTIONS")
    
    # Multiple real customers
    customers = [
        {
            "customer_id": "BATCH-001",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "contract_type": "Month-to-month",
            "PaperlessBilling": "Yes",
            "payment_method": "Mailed check",
            "MonthlyCharges": 53.85,
            "TotalCharges": 108.15,
            "service_combo_id": "DSL_Yes_No_No",
            "geo_code": "G15"
        },
        {
            "customer_id": "BATCH-002",
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 65,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "contract_type": "Two year",
            "PaperlessBilling": "No",
            "payment_method": "Bank transfer (automatic)",
            "MonthlyCharges": 105.50,
            "TotalCharges": 6852.50,
            "service_combo_id": "Fiber optic_Yes_Yes_Yes",
            "geo_code": "G42"
        },
        {
            "customer_id": "BATCH-003",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "contract_type": "One year",
            "PaperlessBilling": "No",
            "payment_method": "Bank transfer (automatic)",
            "MonthlyCharges": 56.15,
            "TotalCharges": 1347.60,
            "service_combo_id": "DSL_Yes_No_No",
            "geo_code": "G8"
        }
    ]
    
    print(f"📤 Testing {len(customers)} customers...")
    
    success_count = 0
    results = []
    
    for i, customer in enumerate(customers, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=customer,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                results.append(data)
                print(f"   ✅ Customer {i} ({customer['customer_id']}): "
                      f"Churn={data.get('churn_probability', 0):.2f}, "
                      f"Risk={data.get('risk_level')}")
                success_count += 1
            else:
                print(f"   ❌ Customer {i} ({customer['customer_id']}): "
                      f"FAILED (status {response.status_code})")
        except Exception as e:
            print(f"   ❌ Customer {i}: ERROR - {e}")
    
    print(f"\n📊 Results: {success_count}/{len(customers)} successful")
    
    if results:
        high_risk = sum(1 for r in results if r.get('risk_level') == 'HIGH')
        medium_risk = sum(1 for r in results if r.get('risk_level') == 'MEDIUM')
        low_risk = sum(1 for r in results if r.get('risk_level') == 'LOW')
        
        print(f"   Risk Distribution:")
        print(f"   - HIGH: {high_risk}")
        print(f"   - MEDIUM: {medium_risk}")
        print(f"   - LOW: {low_risk}")
    
    return success_count == len(customers)


def test_error_handling():
    """Test 5: Error handling"""
    print_section("TEST 5: ERROR HANDLING")
    
    tests_passed = 0
    
    # Test 1: Missing required fields
    print("1️⃣ Testing missing required fields...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"customer_id": "TEST"},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code in [422, 400]:
            print(f"   ✅ Correctly rejected (status {response.status_code})")
            tests_passed += 1
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test 2: Invalid data types
    print("2️⃣ Testing invalid data types...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={
                "customer_id": "TEST",
                "tenure": "not_a_number",  # Should be int
                "MonthlyCharges": "invalid"  # Should be float
            },
            headers={"Content-Type": "application/json"}
        )
        if response.status_code in [422, 400]:
            print(f"   ✅ Correctly rejected (status {response.status_code})")
            tests_passed += 1
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    print(f"\n📊 Error Handling: {tests_passed}/2 tests passed")
    return tests_passed == 2


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 TELCO CHURN API - REAL DATA TESTS")
    print("="*60)
    print(f"🌐 Testing API at: {BASE_URL}")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Run all tests
    results['health_check'] = test_health_check()
    results['model_info'] = test_model_info()
    results['single_prediction'] = test_single_prediction()
    results['batch_predictions'] = test_batch_predictions()
    results['error_handling'] = test_error_handling()
    
    # Summary
    print("\n" + "="*60)
    print("📊 FINAL TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! API IS PRODUCTION READY!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
    
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        exit(1)