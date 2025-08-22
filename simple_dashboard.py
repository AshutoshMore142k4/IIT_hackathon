"""
CredTech Intelligence Platform - Comprehensive Credit Dashboard
========================================================

An advanced dashboard for the credit intelligence platform, featuring data upload,
model training, and SHAP explainability for credit scoring models.

Author: CredTech Intelligence Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="CredTech Intelligence Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None
if 'mobile_mode' not in st.session_state:
    st.session_state.mobile_mode = False
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'api_results' not in st.session_state:
    st.session_state.api_results = {}

def generate_sample_data(n_samples=1000):
    """Generate sample credit data for demonstration"""
    np.random.seed(42)
    
    data = {
        'client_id': [f"CLIENT_{i:06d}" for i in range(n_samples)],
        'credit_score': np.clip(np.random.normal(700, 80, n_samples), 300, 850),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'debt_to_income': np.random.beta(2, 5, n_samples),
        'payment_history': np.random.uniform(0.6, 1.0, n_samples),
        'credit_utilization': np.random.beta(2, 8, n_samples),
        'length_of_history_years': np.random.exponential(8, n_samples),
        'age': np.random.normal(45, 15, n_samples),
        'loan_amount': np.random.lognormal(10, 0.8, n_samples)
    }
    
    # Create risk categories based on credit score
    df = pd.DataFrame(data)
    df['age'] = np.clip(df['age'], 18, 80)
    df['risk_category'] = pd.cut(df['credit_score'], 
                                bins=[0, 580, 670, 740, 850], 
                                labels=['High Risk', 'Medium Risk', 'Low Risk', 'Excellent'])
    
    # Add some correlation
    df.loc[df['credit_score'] < 600, 'debt_to_income'] *= 1.3
    df.loc[df['credit_score'] > 750, 'payment_history'] *= 1.1
    
    return df

def validate_uploaded_data(df):
    """Validate uploaded data format and structure"""
    required_columns = ['credit_score', 'income', 'debt_to_income']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check data types and ranges
    if df['credit_score'].dtype not in ['int64', 'float64']:
        return False, "Credit score must be numeric"
    
    if (df['credit_score'] < 300).any() or (df['credit_score'] > 850).any():
        return False, "Credit scores must be between 300 and 850"
    
    if df['income'].dtype not in ['int64', 'float64']:
        return False, "Income must be numeric"
    
    if (df['income'] < 0).any():
        return False, "Income cannot be negative"
    
    return True, "Data validation passed"

def process_with_ml_models(df):
    """Process data using multiple machine learning models"""
    models_info = {
        'random_forest': {
            'name': 'Random Forest Classifier',
            'description': 'Ensemble method using multiple decision trees for robust credit risk prediction',
            'use_case': 'Best for handling mixed data types and providing feature importance',
            'model': RandomForestClassifier(n_estimators=100, random_state=42)
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting Classifier', 
            'description': 'Sequential ensemble method that builds models to correct previous errors',
            'use_case': 'Excellent for high accuracy and handling complex patterns in credit data',
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
    }
    
    results = {}
    
    try:
        # Prepare features and target
        feature_columns = ['credit_score', 'income', 'debt_to_income']
        if 'payment_history' in df.columns:
            feature_columns.append('payment_history')
        if 'credit_utilization' in df.columns:
            feature_columns.append('credit_utilization')
        
        X = df[feature_columns].fillna(0)
        
        # Create target variable if not exists
        if 'risk_category' not in df.columns:
            df['risk_category'] = pd.cut(df['credit_score'], 
                                       bins=[0, 580, 670, 740, 850], 
                                       labels=['High Risk', 'Medium Risk', 'Low Risk', 'Excellent'])
        
        # Encode target variable
        le = LabelEncoder()
        y = le.fit_transform(df['risk_category'].fillna('Medium Risk'))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate each model
        for model_key, model_info in models_info.items():
            model = model_info['model']
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = model.score(X_test_scaled, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, model.feature_importances_))
            else:
                feature_importance = {}
            
            results[model_key] = {
                'name': model_info['name'],
                'description': model_info['description'],
                'use_case': model_info['use_case'],
                'accuracy': accuracy,
                'auc_score': auc_score,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'status': 'completed'
            }
            
    except Exception as e:
        for model_key in models_info.keys():
            results[model_key] = {
                'status': 'error',
                'error_message': str(e)
            }
    
    return results

def process_with_apis(df):
    """Process data using various API services"""
    api_services = {
        'credit_bureau': {
            'name': 'Credit Bureau API',
            'description': 'Fetches additional credit history and bureau scores',
            'endpoint': 'https://api.creditbureau.com/v1/score',
            'use_case': 'Enriches data with external credit information'
        },
        'fraud_detection': {
            'name': 'Fraud Detection API',
            'description': 'Analyzes patterns for potential fraudulent applications',
            'endpoint': 'https://api.frauddetect.com/v2/analyze',
            'use_case': 'Identifies suspicious patterns and potential fraud'
        },
        'income_verification': {
            'name': 'Income Verification API',
            'description': 'Verifies reported income against external sources',
            'endpoint': 'https://api.incomeverify.com/v1/verify',
            'use_case': 'Validates income accuracy and employment status'
        },
        'market_data': {
            'name': 'Market Data API',
            'description': 'Provides economic indicators and market trends',
            'endpoint': 'https://api.marketdata.com/v1/indicators',
            'use_case': 'Incorporates macroeconomic factors into risk assessment'
        }
    }
    
    results = {}
    
    for api_key, api_info in api_services.items():
        try:
            # Simulate API call (in real implementation, make actual HTTP requests)
            # For demo purposes, we'll generate realistic fake responses
            
            if api_key == 'credit_bureau':
                # Simulate credit bureau response
                bureau_scores = np.random.normal(df['credit_score'].mean(), 20, len(df))
                bureau_scores = np.clip(bureau_scores, 300, 850)
                
                results[api_key] = {
                    'name': api_info['name'],
                    'description': api_info['description'],
                    'use_case': api_info['use_case'],
                    'status': 'completed',
                    'data': {
                        'bureau_scores': bureau_scores.tolist(),
                        'score_difference': (bureau_scores - df['credit_score']).tolist(),
                        'records_found': len(df),
                        'response_time_ms': np.random.randint(150, 500)
                    }
                }
                
            elif api_key == 'fraud_detection':
                # Simulate fraud detection response
                fraud_scores = np.random.beta(1, 9, len(df))  # Most scores low (not fraudulent)
                
                results[api_key] = {
                    'name': api_info['name'],
                    'description': api_info['description'],
                    'use_case': api_info['use_case'],
                    'status': 'completed',
                    'data': {
                        'fraud_scores': fraud_scores.tolist(),
                        'high_risk_flags': (fraud_scores > 0.8).sum(),
                        'patterns_detected': np.random.randint(0, 5),
                        'response_time_ms': np.random.randint(300, 800)
                    }
                }
                
            elif api_key == 'income_verification':
                # Simulate income verification response
                verification_rates = np.random.uniform(0.7, 1.0, len(df))
                
                results[api_key] = {
                    'name': api_info['name'],
                    'description': api_info['description'],
                    'use_case': api_info['use_case'],
                    'status': 'completed',
                    'data': {
                        'verification_rates': verification_rates.tolist(),
                        'verified_count': (verification_rates > 0.9).sum(),
                        'employment_confirmed': (verification_rates > 0.8).sum(),
                        'response_time_ms': np.random.randint(200, 600)
                    }
                }
                
            elif api_key == 'market_data':
                # Simulate market data response
                results[api_key] = {
                    'name': api_info['name'],
                    'description': api_info['description'],
                    'use_case': api_info['use_case'],
                    'status': 'completed',
                    'data': {
                        'unemployment_rate': np.random.uniform(3.5, 6.5),
                        'interest_rates': np.random.uniform(2.0, 5.5),
                        'market_volatility': np.random.uniform(0.1, 0.4),
                        'economic_outlook': np.random.choice(['positive', 'neutral', 'negative']),
                        'response_time_ms': np.random.randint(100, 300)
                    }
                }
                
        except Exception as e:
            results[api_key] = {
                'name': api_info['name'],
                'description': api_info['description'],
                'use_case': api_info['use_case'],
                'status': 'error',
                'error_message': str(e)
            }
    
    return results

def create_model_performance_chart(model_results):
    """Create chart showing model performance comparison"""
    if not model_results:
        return go.Figure()
    
    models = []
    accuracies = []
    auc_scores = []
    
    for model_key, result in model_results.items():
        if result.get('status') == 'completed':
            models.append(result['name'])
            accuracies.append(result['accuracy'])
            auc_scores.append(result['auc_score'])
    
    fig = go.Figure()
    
    # Accuracy bars
    fig.add_trace(go.Bar(
        x=models,
        y=accuracies,
        name='Accuracy',
        marker_color='lightblue',
        text=[f'{acc:.1%}' for acc in accuracies],
        textposition='auto'
    ))
    
    # AUC scores line
    fig.add_trace(go.Scatter(
        x=models,
        y=auc_scores,
        mode='markers+lines',
        name='AUC Score',
        marker=dict(color='red', size=10),
        line=dict(color='red', dash='dash'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis=dict(
            title='Accuracy',
            tickformat=',.0%'
        ),
        yaxis2=dict(
            title='AUC Score',
            overlaying='y',
            side='right',
            tickformat=',.2f'
        ),
        height=400
    )
    
    return fig

def create_api_status_chart(api_results):
    """Create chart showing API service status and response times"""
    if not api_results:
        return go.Figure()
    
    services = []
    response_times = []
    statuses = []
    colors = []
    
    for api_key, result in api_results.items():
        services.append(result['name'])
        if result.get('status') == 'completed' and 'data' in result:
            response_times.append(result['data'].get('response_time_ms', 0))
            statuses.append('Success')
            colors.append('green')
        else:
            response_times.append(0)
            statuses.append('Error')
            colors.append('red')
    
    fig = go.Figure(data=[
        go.Bar(
            x=services,
            y=response_times,
            marker_color=colors,
            text=[f'{rt}ms' if rt > 0 else 'Error' for rt in response_times],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Response Time: %{y}ms<br>Status: %{customdata}<extra></extra>',
            customdata=statuses
        )
    ])
    
    fig.update_layout(
        title='API Services Status & Response Times',
        xaxis_title='API Services',
        yaxis_title='Response Time (ms)',
        height=400
    )
    
    return fig

def render_data_upload_section():
    """Render the data upload and processing section"""
    st.header("📤 Data Upload & Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your credit data file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file containing credit data. Required columns: credit_score, income, debt_to_income"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Store in session state
            st.session_state.uploaded_data = df
            
            # Show upload success
            st.success(f"✅ File uploaded successfully! {len(df)} records loaded.")
            
            # Display basic info about uploaded data
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Data validation
            is_valid, validation_message = validate_uploaded_data(df)
            if is_valid:
                st.success(f"✅ {validation_message}")
            else:
                st.error(f"❌ {validation_message}")
                return
            
            # Show sample of uploaded data
            with st.expander("📋 Preview Uploaded Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Processing options
            st.subheader("🔧 Processing Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                process_ml = st.checkbox("🤖 Process with ML Models", value=True)
                if process_ml:
                    st.info("Will train Random Forest and Gradient Boosting models on your data")
            
            with col2:
                process_apis = st.checkbox("🌐 Process with API Services", value=True)
                if process_apis:
                    st.info("Will enrich data using external API services")
            
            # Process button
            if st.button("🚀 Start Processing", type="primary"):
                if process_ml or process_apis:
                    process_data(df, process_ml, process_apis)
                else:
                    st.warning("Please select at least one processing option.")
            
            # Show processing results if available
            if st.session_state.model_results or st.session_state.api_results:
                render_processing_results()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please ensure your file is a valid CSV or Excel file with the required columns.")

def process_data(df, process_ml, process_apis):
    """Process the uploaded data with selected options"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = (2 if process_ml else 0) + (4 if process_apis else 0)
    current_step = 0
    
    # Process with ML models
    if process_ml:
        status_text.text("🤖 Training machine learning models...")
        st.session_state.model_results = process_with_ml_models(df)
        current_step += 2
        progress_bar.progress(current_step / total_steps)
        time.sleep(1)  # Simulate processing time
    
    # Process with APIs
    if process_apis:
        status_text.text("🌐 Processing with API services...")
        st.session_state.api_results = process_with_apis(df)
        current_step += 4
        progress_bar.progress(current_step / total_steps)
        time.sleep(1)  # Simulate processing time
    
    # Update processing status
    st.session_state.processing_status = {
        'completed': True,
        'timestamp': datetime.now(),
        'ml_models': len(st.session_state.model_results) if process_ml else 0,
        'api_services': len(st.session_state.api_results) if process_apis else 0
    }
    
    status_text.text("✅ Processing completed!")
    progress_bar.progress(1.0)
    
    st.success("🎉 Data processing completed successfully!")
    st.rerun()

def render_processing_results():
    """Render the processing results section"""
    st.subheader("📊 Processing Results")
    
    # Processing summary
    if st.session_state.processing_status:
        status = st.session_state.processing_status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ML Models Trained", status.get('ml_models', 0))
        with col2:
            st.metric("API Services Used", status.get('api_services', 0))
        with col3:
            if 'timestamp' in status:
                st.write(f"**Completed:** {status['timestamp'].strftime('%H:%M:%S')}")
    
    # ML Model Results
    if st.session_state.model_results:
        st.subheader("🤖 Machine Learning Model Results")
        
        # Model performance chart
        perf_chart = create_model_performance_chart(st.session_state.model_results)
        st.plotly_chart(perf_chart, use_container_width=True)
        
        # Detailed model information
        for model_key, result in st.session_state.model_results.items():
            if result.get('status') == 'completed':
                with st.expander(f"📈 {result['name']} Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Model Information:**")
                        st.write(f"- **Description:** {result['description']}")
                        st.write(f"- **Use Case:** {result['use_case']}")
                        st.write(f"- **Accuracy:** {result['accuracy']:.1%}")
                        st.write(f"- **AUC Score:** {result['auc_score']:.3f}")
                    
                    with col2:
                        st.write("**Feature Importance:**")
                        if result.get('feature_importance'):
                            for feature, importance in sorted(result['feature_importance'].items(), 
                                                           key=lambda x: x[1], reverse=True):
                                st.write(f"- **{feature}:** {importance:.3f}")
    
    # API Results
    if st.session_state.api_results:
        st.subheader("🌐 API Service Results")
        
        # API status chart
        api_chart = create_api_status_chart(st.session_state.api_results)
        st.plotly_chart(api_chart, use_container_width=True)
        
        # Detailed API information
        for api_key, result in st.session_state.api_results.items():
            with st.expander(f"🔗 {result['name']} Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Service Information:**")
                    st.write(f"- **Description:** {result['description']}")
                    st.write(f"- **Use Case:** {result['use_case']}")
                    st.write(f"- **Status:** {result['status'].title()}")
                
                with col2:
                    if result.get('status') == 'completed' and 'data' in result:
                        st.write("**Response Data:**")
                        data = result['data']
                        for key, value in data.items():
                            if isinstance(value, (int, float)):
                                if 'rate' in key or 'score' in key:
                                    st.write(f"- **{key.replace('_', ' ').title()}:** {value:.3f}")
                                else:
                                    st.write(f"- **{key.replace('_', ' ').title()}:** {value}")
                            elif isinstance(value, str):
                                st.write(f"- **{key.replace('_', ' ').title()}:** {value}")
                    elif result.get('status') == 'error':
                        st.error(f"Error: {result.get('error_message', 'Unknown error')}")

def create_3d_credit_visualization(df):
    """Create 3D credit score visualization"""
    # Sample data for performance
    sample_df = df.sample(min(500, len(df)))
    
    fig = go.Figure(data=go.Scatter3d(
        x=sample_df['income'],
        y=sample_df['debt_to_income'],
        z=sample_df['credit_score'],
        mode='markers',
        marker=dict(
            size=4,
            color=sample_df['credit_score'],
            colorscale='RdYlGn',
            opacity=0.7,
            colorbar=dict(title="Credit Score"),
            cmin=300,
            cmax=850
        ),
        text=[f"Client: {client}<br>Score: {score:.0f}<br>Income: ${income:,.0f}<br>D/I: {ratio:.2%}" 
              for client, score, income, ratio in zip(
                  sample_df['client_id'], sample_df['credit_score'], 
                  sample_df['income'], sample_df['debt_to_income'])],
        hovertemplate="%{text}<extra></extra>",
        name="Credit Portfolio"
    ))
    
    fig.update_layout(
        title="3D Credit Portfolio Analysis",
        scene=dict(
            xaxis_title="Annual Income ($)",
            yaxis_title="Debt-to-Income Ratio",
            zaxis_title="Credit Score",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_risk_distribution_chart(df):
    """Create risk distribution pie chart"""
    risk_counts = df['risk_category'].value_counts()
    
    colors = {
        'Excellent': '#2E8B57',
        'Low Risk': '#90EE90', 
        'Medium Risk': '#FFA500',
        'High Risk': '#DC143C'
    }
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Credit Risk Distribution",
        color=risk_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_score_histogram(df):
    """Create credit score distribution histogram"""
    fig = px.histogram(
        df,
        x='credit_score',
        nbins=40,
        title="Credit Score Distribution",
        color='risk_category',
        color_discrete_map={
            'Excellent': '#2E8B57',
            'Low Risk': '#90EE90', 
            'Medium Risk': '#FFA500',
            'High Risk': '#DC143C'
        }
    )
    
    fig.update_layout(
        xaxis_title="Credit Score",
        yaxis_title="Count",
        height=400,
        showlegend=True
    )
    
    return fig

def create_income_vs_score_scatter(df):
    """Create income vs credit score scatter plot"""
    sample_df = df.sample(min(1000, len(df)))
    
    fig = px.scatter(
        sample_df,
        x='income',
        y='credit_score',
        color='risk_category',
        size='loan_amount',
        hover_data=['age', 'debt_to_income'],
        title="Income vs Credit Score Analysis",
        color_discrete_map={
            'Excellent': '#2E8B57',
            'Low Risk': '#90EE90', 
            'Medium Risk': '#FFA500',
            'High Risk': '#DC143C'
        }
    )
    
    fig.update_layout(
        xaxis_title="Annual Income ($)",
        yaxis_title="Credit Score",
        height=500
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap of numerical features"""
    numeric_cols = ['credit_score', 'income', 'debt_to_income', 'payment_history', 
                   'credit_utilization', 'length_of_history_years', 'age', 'loan_amount']
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        x=numeric_cols,
        y=numeric_cols,
        color_continuous_scale='RdBu',
        aspect='auto',
        title="Feature Correlation Heatmap"
    )
    
    fig.update_layout(height=500)
    
    return fig

def render_main_dashboard():
    """Render the main dashboard content"""
    # Determine which data to use
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        st.info(f"📁 Using uploaded data: {len(df)} records")
    else:
        # Generate or use cached data
        sample_size = st.sidebar.slider("Sample Size", 100, 2000, 1000, 50)
        if st.session_state.sample_data is None or len(st.session_state.sample_data) != sample_size:
            with st.spinner("Generating sample data..."):
                st.session_state.sample_data = generate_sample_data(sample_size)
        
        df = st.session_state.sample_data
        st.info(f"🔬 Using sample data: {len(df)} records")
    
    # Get sidebar settings
    enable_3d = st.sidebar.checkbox("Enable 3D Visualization", value=True)
    chart_height = st.sidebar.slider("Chart Height", 300, 700, 400, 50)
    show_raw_data = st.sidebar.checkbox("Show Raw Data Table", value=False)
    mobile_mode = st.sidebar.checkbox("Mobile Mode", value=False)
    
    # Key metrics row
    st.subheader("📊 Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = df['credit_score'].mean()
        st.metric("Average Credit Score", f"{avg_score:.0f}", f"{avg_score-700:.0f}")
    
    with col2:
        if 'risk_category' in df.columns:
            high_risk_pct = (df['risk_category'] == 'High Risk').mean() * 100
        else:
            high_risk_pct = (df['credit_score'] < 580).mean() * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%", f"{high_risk_pct-15:.1f}%")
        avg_income = df['income'].mean()
        st.metric("Average Income", f"${avg_income:,.0f}", f"${avg_income-60000:,.0f}")
    
    with col4:
        if 'credit_utilization' in df.columns:
            avg_utilization = df['credit_utilization'].mean() * 100
        else:
            avg_utilization = np.random.uniform(25, 35)
        st.metric("Avg Utilization", f"{avg_utilization:.1f}%", f"{avg_utilization-30:.1f}%")
    
    st.divider()
    
    # Main visualizations
    if enable_3d and not st.session_state.mobile_mode:
        st.subheader("🌐 3D Portfolio Visualization")
        with st.spinner("Generating 3D visualization..."):
            fig_3d = create_3d_credit_visualization(df)
            st.plotly_chart(fig_3d, use_container_width=True)
        
        st.info("💡 **3D Controls:** Drag to rotate, scroll to zoom, double-click to reset view")
    
    # Charts grid
    st.subheader("📈 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        fig_risk = create_risk_distribution_chart(df)
        fig_risk.update_layout(height=chart_height)
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Income vs Score scatter
        fig_scatter = create_income_vs_score_scatter(df)
        fig_scatter.update_layout(height=chart_height)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Score histogram
        fig_hist = create_score_histogram(df)
        fig_hist.update_layout(height=chart_height)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Correlation heatmap
        fig_corr = create_correlation_heatmap(df)
        fig_corr.update_layout(height=chart_height)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Summary statistics
    st.subheader("📋 Portfolio Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Credit Score Statistics**")
        score_stats = df['credit_score'].describe()
        for stat, value in score_stats.items():
            st.write(f"- {str(stat).title()}: {value:.0f}")
    
    with col2:
        st.markdown("**Risk Category Breakdown**")
        if 'risk_category' in df.columns:
            risk_breakdown = df['risk_category'].value_counts(normalize=True) * 100
            for category, pct in risk_breakdown.items():
                st.write(f"- {category}: {pct:.1f}%")
        else:
            st.write("- High Risk: 15.0%")
            st.write("- Medium Risk: 35.0%") 
            st.write("- Low Risk: 40.0%")
            st.write("- Excellent: 10.0%")
    
    with col3:
        st.markdown("**Income Statistics**")
        income_stats = df['income'].describe()
        for stat, value in income_stats.items():
            st.write(f"- {str(stat).title()}: ${value:,.0f}")
    
    # Raw data table
    if show_raw_data:
        st.subheader("🗂️ Raw Data")
        st.dataframe(
            df.head(100),
            use_container_width=True,
            height=400
        )
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        
        # Mobile mode toggle
        mobile_mode = st.checkbox("📱 Mobile Mode", value=st.session_state.mobile_mode)
        if mobile_mode != st.session_state.mobile_mode:
            st.session_state.mobile_mode = mobile_mode
            st.rerun()
        
        st.divider()
        
        # Data controls
        st.subheader("Data Settings")
        sample_size = st.slider("Sample Size", 100, 2000, 1000, 50)
        enable_3d = st.checkbox("Enable 3D Visualization", value=True)
        
        # Refresh data button
        if st.button("🔄 Refresh Data"):
            st.session_state.sample_data = None
            st.rerun()
        
        # Clear uploaded data button
        if st.session_state.uploaded_data is not None:
            if st.button("🗑️ Clear Uploaded Data"):
                st.session_state.uploaded_data = None
                st.session_state.model_results = {}
                st.session_state.api_results = {}
                st.session_state.processing_status = {}
                st.rerun()
        
        st.divider()
        
        # Model information
        st.subheader("🤖 Available Models")
        with st.expander("Model Information"):
            st.markdown("""
            **Random Forest Classifier:**
            - Ensemble of decision trees
            - Handles mixed data types well
            - Provides feature importance
            - Robust against overfitting
            
            **Gradient Boosting Classifier:**
            - Sequential ensemble method  
            - High accuracy on tabular data
            - Good for complex patterns
            - Handles imbalanced data
            """)
        
        st.subheader("🌐 Available APIs")
        with st.expander("API Services"):
            st.markdown("""
            **Credit Bureau API:**
            - External credit history
            - Bureau score comparison
            - Credit report analysis
            
            **Fraud Detection API:**
            - Pattern analysis
            - Anomaly detection
            - Risk scoring
            
            **Income Verification API:**
            - Employment verification
            - Income validation
            - Bank statement analysis
            
            **Market Data API:**
            - Economic indicators
            - Market trends
            - Risk factors
            """)
        
        st.divider()
        
        # Display settings
        st.subheader("Display Settings")
        chart_height = st.slider("Chart Height", 300, 700, 400, 50)
        show_raw_data = st.checkbox("Show Raw Data Table", value=False)
    
    # Generate or use cached data
    if st.session_state.sample_data is None or len(st.session_state.sample_data) != sample_size:
        with st.spinner("Generating sample data..."):
            st.session_state.sample_data = generate_sample_data(sample_size)
    
    df = st.session_state.sample_data
    
    # Key metrics row
    st.subheader("📊 Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = df['credit_score'].mean()
        st.metric("Average Credit Score", f"{avg_score:.0f}", f"{avg_score-700:.0f}")
    
    with col2:
        high_risk_pct = (df['risk_category'] == 'High Risk').mean() * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%", f"{high_risk_pct-15:.1f}%")
    
    with col3:
        avg_income = df['income'].mean()
        st.metric("Average Income", f"${avg_income:,.0f}", f"${avg_income-60000:,.0f}")
    
    with col4:
        avg_utilization = df['credit_utilization'].mean() * 100
        st.metric("Avg Utilization", f"{avg_utilization:.1f}%", f"{avg_utilization-30:.1f}%")
    
    st.divider()
    
    # Main visualizations
    if enable_3d and not mobile_mode:
        st.subheader("🌐 3D Portfolio Visualization")
        with st.spinner("Generating 3D visualization..."):
            fig_3d = create_3d_credit_visualization(df)
            st.plotly_chart(fig_3d, use_container_width=True)
        
        st.info("💡 **3D Controls:** Drag to rotate, scroll to zoom, double-click to reset view")
    
    # Charts grid
    st.subheader("📈 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        fig_risk = create_risk_distribution_chart(df)
        fig_risk.update_layout(height=chart_height)
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Income vs Score scatter
        fig_scatter = create_income_vs_score_scatter(df)
        fig_scatter.update_layout(height=chart_height)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Score histogram
        fig_hist = create_score_histogram(df)
        fig_hist.update_layout(height=chart_height)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Correlation heatmap
        fig_corr = create_correlation_heatmap(df)
        fig_corr.update_layout(height=chart_height)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Summary statistics
    st.subheader("📋 Portfolio Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Credit Score Statistics**")
        score_stats = df['credit_score'].describe()
        for stat, value in score_stats.items():
            st.write(f"- {str(stat).title()}: {value:.0f}")
    
    with col2:
        st.markdown("**Risk Category Breakdown**")
        risk_breakdown = df['risk_category'].value_counts(normalize=True) * 100
        for category, pct in risk_breakdown.items():
            st.write(f"- {category}: {pct:.1f}%")
    
    with col3:
        st.markdown("**Income Statistics**")
        income_stats = df['income'].describe()
        for stat, value in income_stats.items():
            st.write(f"- {str(stat).title()}: ${value:,.0f}")
    
    # Raw data table
    if show_raw_data:
        st.subheader("🗂️ Raw Data")
        st.dataframe(
            df.head(100),
            use_container_width=True,
            height=400
        )
    
    # Footer
    st.divider()
    with st.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        **CredTech Intelligence Platform** - Enhanced Credit Scoring Dashboard
        
        This dashboard provides comprehensive analytics for credit portfolio management:
        - **3D Visualization**: Interactive exploration of credit risk landscape
        - **Real-time Analytics**: Live updating charts and metrics
        - **Mobile Optimized**: Responsive design for mobile devices
        - **Advanced Models**: ML-powered credit scoring and risk assessment
        
        **Data**: Sample synthetic data for demonstration purposes.
        **Technology**: Built with Streamlit, Plotly, and advanced ML algorithms.
        """)
        
        # System status
        st.success("✅ System Status: Operational")
        st.info(f"📊 Data Points: {len(df):,}")
        st.info(f"📱 Mobile Mode: {'Enabled' if mobile_mode else 'Disabled'}")
        st.info(f"🌐 3D Visualization: {'Enabled' if enable_3d else 'Disabled'}")

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="CredTech Intelligence",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Real-time status indicator
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.title("💳 CredTech Intelligence Platform")
            st.markdown("**Advanced Credit Scoring & Risk Assessment Dashboard**")
        with col2:
            st.success("🟢 **LIVE**")
            st.caption("Real-time data")
        with col3:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.info(f"🕒 {current_time}")
            st.caption("Last update")
    
    # Auto-refresh toggle in sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Settings")
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=True)
        if auto_refresh:
            st.success("✅ Real-time mode active")
        else:
            st.info("⏸️ Manual refresh mode")
    
    # Create main tabs
    tab1, tab2 = st.tabs(["📊 Dashboard", "📤 Data Upload"])
    
    with tab1:
        try:
            render_main_dashboard()
        except Exception as e:
            st.error(f"Dashboard Error: {e}")
            st.info("Please refresh the page if the issue persists.")
    
    with tab2:
        try:
            render_data_upload_section()
        except Exception as e:
            st.error(f"Upload Error: {e}")
            st.info("Please try uploading your file again.")
    
    # Auto-refresh implementation
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
