"""
Smoke Test - End-to-End Deployment Verification
Tests that the deployed service is up and responding
"""
import requests
import time
import sys

def smoke_test_api():
    """
    Smoke test: Verify API is running and healthy
    This is an END-TO-END test that verifies:
    1. Container is running
    2. API service is up
    3. Health endpoint responds
    4. Model is loaded
    """
    
    api_url = "http://localhost:8000"
    max_retries = 5
    retry_delay = 2
    
    print("🔥 Starting Smoke Test...")
    
    # Wait for service to be ready
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Checking health...")
            response = requests.get(f"{api_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                
                # Verify model is loaded
                if data.get("model_loaded") == True:
                    print("✅ Model loaded successfully")
                    print("✅ SMOKE TEST PASSED!")
                    return 0
                else:
                    print("❌ Model not loaded")
                    return 1
            else:
                print(f"⚠️ Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Connection failed: {e}")
        
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    print("❌ SMOKE TEST FAILED: Service not responding")
    return 1

if __name__ == "__main__":
    exit_code = smoke_test_api()
    sys.exit(exit_code)