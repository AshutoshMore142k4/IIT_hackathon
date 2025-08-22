"""
Test script to verify SHAP integration
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap

def test_shap_integration():
    print("Testing SHAP integration...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'credit_score': np.clip(np.random.normal(700, 80, n_samples), 300, 850),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'debt_to_income': np.random.beta(2, 5, n_samples),
        'age': np.clip(np.random.normal(45, 15, n_samples), 18, 80),
        'loan_amount': np.random.lognormal(10, 0.8, n_samples)
    }
    
    df = pd.DataFrame(data)
    df['risk_category'] = pd.cut(df['credit_score'], 
                                bins=[0, 580, 670, 740, 850], 
                                labels=['High Risk', 'Medium Risk', 'Low Risk', 'Excellent'])
    
    print(f"✅ Generated {len(df)} sample records")
    
    # Train model
    feature_columns = ['credit_score', 'income', 'debt_to_income', 'age', 'loan_amount']
    X = df[feature_columns]
    
    le = LabelEncoder()
    y = le.fit_transform(df['risk_category'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    print("✅ Model trained successfully")
    
    # Test SHAP
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.iloc[:5])  # Test with first 5 samples
        
        print("✅ SHAP explainer created successfully")
        print(f"✅ SHAP values shape: {np.array(shap_values).shape}")
        print("✅ SHAP integration test passed!")
        
        return True
    except Exception as e:
        print(f"❌ SHAP integration failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_shap_integration()
    if success:
        print("\n🎉 All tests passed! SHAP is properly integrated.")
    else:
        print("\n💥 Tests failed. Check the error messages above.")
