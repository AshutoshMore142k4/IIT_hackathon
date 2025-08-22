"""
Test script for Enhanced Dashboard
Validates all features including 3D visualizations and SHAP
"""

import sys
sys.path.append('.')

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas and NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Pandas/NumPy import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import shap
        print(f"✅ SHAP imported successfully (version: {shap.__version__})")
    except ImportError as e:
        print(f"❌ SHAP import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        print("✅ Seaborn and Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Seaborn/Matplotlib import failed: {e}")
        return False
    
    return True

def test_dashboard_functions():
    """Test key dashboard functions"""
    print("\n🧪 Testing dashboard functions...")
    
    try:
        # Import the functions from enhanced_dashboard
        from enhanced_dashboard import (
            generate_enhanced_sample_data,
            create_3d_portfolio_visualization,
            create_risk_heatmap,
            create_pca_visualization,
            create_clustering_analysis,
            create_model_comparison_dashboard
        )
        print("✅ All dashboard functions imported successfully")
        
        # Test data generation
        df = generate_enhanced_sample_data(100)
        print(f"✅ Sample data generated: {len(df)} records")
        print(f"   - Features: {list(df.columns)}")
        print(f"   - Risk categories: {df['risk_category'].value_counts().to_dict()}")
        
        # Test visualizations
        fig_3d = create_3d_portfolio_visualization(df)
        print("✅ 3D visualization created successfully")
        
        fig_heatmap = create_risk_heatmap(df)
        print("✅ Risk heatmap created successfully")
        
        fig_pca = create_pca_visualization(df)
        print("✅ PCA visualization created successfully")
        
        fig_cluster = create_clustering_analysis(df)
        print("✅ Clustering analysis created successfully")
        
        # Test model comparison
        results = create_model_comparison_dashboard(df)
        print(f"✅ Model comparison completed: {list(results.keys())}")
        for model_name, result in results.items():
            print(f"   - {model_name}: {result['accuracy']:.1%} accuracy")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shap_functionality():
    """Test SHAP explainability features"""
    print("\n🧪 Testing SHAP functionality...")
    
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from enhanced_dashboard import generate_enhanced_sample_data
        
        # Generate test data
        df = generate_enhanced_sample_data(200)
        feature_columns = ['credit_score', 'income', 'debt_to_income', 'payment_history', 'credit_utilization']
        X = df[feature_columns].fillna(0)
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df['risk_category'].fillna('Medium Risk'))
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:5])  # Test with 5 samples
        
        print("✅ SHAP TreeExplainer created successfully")
        print(f"   - SHAP values shape: {len(shap_values)} classes")
        print(f"   - Each class shape: {shap_values[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ SHAP functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_3d_visualization():
    """Test 3D visualization capabilities"""
    print("\n🧪 Testing 3D visualization...")
    
    try:
        import plotly.graph_objects as go
        from enhanced_dashboard import generate_enhanced_sample_data, create_3d_portfolio_visualization
        
        # Generate test data
        df = generate_enhanced_sample_data(50)
        
        # Create 3D visualization
        fig = create_3d_portfolio_visualization(df)
        
        # Validate figure structure
        assert len(fig.data) > 0, "No data traces in 3D plot"
        assert fig.data[0].type == 'scatter3d', "First trace is not scatter3d"
        
        print("✅ 3D visualization test passed")
        print(f"   - Data points: {len(fig.data[0].x)}")
        print(f"   - Plot type: {fig.data[0].type}")
        
        return True
        
    except Exception as e:
        print(f"❌ 3D visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Enhanced Dashboard Validation Test")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Dashboard Functions", test_dashboard_functions),
        ("SHAP Functionality", test_shap_functionality),
        ("3D Visualization", test_3d_visualization)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Dashboard is ready for deployment.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
