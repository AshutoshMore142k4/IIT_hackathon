"""
CredTech Intelligence Platform - Enhanced Dashboard
=================================================

An advanced credit dashboard with 3D visualizations, comprehensive analytics,
and SHAP explainability features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import time

# Try to import SHAP, but make it optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="CredTech Advanced Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .feature-highlight {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #b8e6f1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
for key in ['model', 'X_test', 'y_test', 'report', 'classes', 'shap_explainer', 'shap_values', 
           'pca_data', 'cluster_data', 'models_comparison', 'feature_data']:
    if key not in st.session_state:
        st.session_state[key] = None

def validate_and_process_uploaded_data(df):
    """Validate and process uploaded data to match expected format"""
    try:
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Common column mappings (case insensitive)
        column_mappings = {
            # Credit Score variations
            'credit_score': ['credit_score', 'creditscore', 'score', 'credit_rating', 'fico_score', 'fico'],
            # Income variations
            'income': ['income', 'annual_income', 'yearly_income', 'salary', 'earnings'],
            # Debt to Income variations
            'debt_to_income': ['debt_to_income', 'debt_income_ratio', 'dti', 'debt_ratio', 'debtincome'],
            # Age variations
            'age': ['age', 'customer_age', 'client_age', 'borrower_age'],
            # Loan amount variations
            'loan_amount': ['loan_amount', 'loan_amt', 'amount', 'principal', 'requested_amount'],
            # Payment history variations
            'payment_history': ['payment_history', 'payment_hist', 'pay_history', 'payment_record'],
            # Credit utilization variations
            'credit_utilization': ['credit_utilization', 'utilization', 'credit_util', 'util_ratio'],
            # Client ID variations
            'client_id': ['client_id', 'customer_id', 'id', 'client_number', 'account_id']
        }
        
        # Apply column mappings
        df_columns_lower = [col.lower().replace(' ', '_') for col in processed_df.columns]
        
        for target_col, possible_names in column_mappings.items():
            found = False
            for possible_name in possible_names:
                if possible_name.lower() in df_columns_lower:
                    original_col = processed_df.columns[df_columns_lower.index(possible_name.lower())]
                    if original_col != target_col:
                        processed_df = processed_df.rename(columns={original_col: target_col})
                    found = True
                    break
            
            # If column not found, create it based on other columns or set defaults
            if not found:
                if target_col == 'credit_score':
                    # Try to create credit score from available data
                    if 'income' in processed_df.columns and 'debt_to_income' in processed_df.columns:
                        # Create synthetic credit score based on income and debt ratio
                        processed_df['credit_score'] = np.clip(
                            650 + (processed_df['income'] / 1000) * 0.5 - (processed_df['debt_to_income'] * 200),
                            300, 850
                        )
                    else:
                        # Default credit scores
                        processed_df['credit_score'] = np.random.normal(700, 80, len(processed_df))
                        processed_df['credit_score'] = np.clip(processed_df['credit_score'], 300, 850)
                
                elif target_col == 'income':
                    if 'credit_score' in processed_df.columns:
                        # Estimate income from credit score
                        processed_df['income'] = np.exp(10 + (processed_df['credit_score'] - 700) * 0.01)
                    else:
                        processed_df['income'] = np.random.lognormal(10.5, 0.5, len(processed_df))
                
                elif target_col == 'debt_to_income':
                    processed_df['debt_to_income'] = np.random.beta(2, 5, len(processed_df))
                
                elif target_col == 'age':
                    processed_df['age'] = np.clip(np.random.normal(45, 15, len(processed_df)), 18, 80)
                
                elif target_col == 'loan_amount':
                    if 'income' in processed_df.columns:
                        processed_df['loan_amount'] = processed_df['income'] * np.random.uniform(0.2, 2.0, len(processed_df))
                    else:
                        processed_df['loan_amount'] = np.random.lognormal(10, 0.8, len(processed_df))
                
                elif target_col == 'payment_history':
                    if 'credit_score' in processed_df.columns:
                        processed_df['payment_history'] = 0.6 + (processed_df['credit_score'] - 300) / (850 - 300) * 0.4
                    else:
                        processed_df['payment_history'] = np.random.uniform(0.6, 1.0, len(processed_df))
                
                elif target_col == 'credit_utilization':
                    processed_df['credit_utilization'] = np.random.beta(2, 8, len(processed_df))
                
                elif target_col == 'client_id':
                    processed_df['client_id'] = [f"CLIENT_{i:06d}" for i in range(len(processed_df))]
        
        # Ensure data types are correct
        numeric_columns = ['credit_score', 'income', 'debt_to_income', 'age', 'loan_amount', 'payment_history', 'credit_utilization']
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        
        # Create risk categories if not present
        if 'risk_category' not in processed_df.columns:
            processed_df['risk_category'] = pd.cut(processed_df['credit_score'], 
                                                  bins=[0, 580, 670, 740, 850], 
                                                  labels=['High Risk', 'Medium Risk', 'Low Risk', 'Excellent'])
        
        # Add additional synthetic features for enhanced analysis
        if 'employment_length' not in processed_df.columns:
            processed_df['employment_length'] = np.random.exponential(5, len(processed_df))
        
        if 'num_accounts' not in processed_df.columns:
            processed_df['num_accounts'] = np.random.poisson(8, len(processed_df))
        
        if 'recent_inquiries' not in processed_df.columns:
            processed_df['recent_inquiries'] = np.random.poisson(2, len(processed_df))
        
        if 'delinquencies_2yrs' not in processed_df.columns:
            processed_df['delinquencies_2yrs'] = np.random.poisson(0.5, len(processed_df))
        
        if 'total_credit_limit' not in processed_df.columns:
            processed_df['total_credit_limit'] = processed_df['income'] * np.random.uniform(0.5, 3.0, len(processed_df))
        
        if 'monthly_payment' not in processed_df.columns:
            processed_df['monthly_payment'] = processed_df['loan_amount'] / np.random.uniform(12, 60, len(processed_df))
        
        if 'debt_consolidation' not in processed_df.columns:
            processed_df['debt_consolidation'] = np.random.choice([0, 1], len(processed_df), p=[0.7, 0.3])
        
        if 'home_ownership' not in processed_df.columns:
            processed_df['home_ownership'] = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], len(processed_df), p=[0.4, 0.3, 0.3])
        
        if 'verification_status' not in processed_df.columns:
            processed_df['verification_status'] = np.random.choice(['Verified', 'Not Verified', 'Source Verified'], 
                                                                  len(processed_df), p=[0.4, 0.3, 0.3])
        
        if 'purpose' not in processed_df.columns:
            processed_df['purpose'] = np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 
                                                       'major_purchase', 'small_business'], len(processed_df), p=[0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Add credit age if missing
        if 'credit_age' not in processed_df.columns:
            processed_df['credit_age'] = processed_df['employment_length'] * np.random.uniform(0.5, 2.0, len(processed_df))
        
        return processed_df, True, "Data processed successfully"
        
    except Exception as e:
        return df, False, f"Error processing data: {str(e)}"

def detect_and_map_columns(df):
    """Intelligently detect and map column names from uploaded data"""
    column_mapping = {}
    
    # Common variations for credit score
    credit_score_variants = ['credit_score', 'creditscore', 'score', 'credit_rating', 'rating', 'fico_score', 'fico', 'credit']
    for variant in credit_score_variants:
        matches = [col for col in df.columns if variant.lower() in col.lower()]
        if matches:
            column_mapping['credit_score'] = matches[0]
            break
    
    # Common variations for income
    income_variants = ['income', 'annual_income', 'salary', 'earnings', 'gross_income', 'net_income', 'pay']
    for variant in income_variants:
        matches = [col for col in df.columns if variant.lower() in col.lower()]
        if matches:
            column_mapping['income'] = matches[0]
            break
    
    # Common variations for debt to income
    dti_variants = ['debt_to_income', 'debt_income_ratio', 'dti', 'debt_ratio', 'debt_income', 'debt_to_income_ratio']
    for variant in dti_variants:
        matches = [col for col in df.columns if variant.lower() in col.lower()]
        if matches:
            column_mapping['debt_to_income'] = matches[0]
            break
    
    # Common variations for age
    age_variants = ['age', 'customer_age', 'client_age', 'years_old', 'age_years']
    for variant in age_variants:
        matches = [col for col in df.columns if variant.lower() in col.lower()]
        if matches:
            column_mapping['age'] = matches[0]
            break
    
    # Common variations for credit utilization
    util_variants = ['credit_utilization', 'utilization', 'util_rate', 'credit_usage', 'balance_ratio', 'credit_limit_used']
    for variant in util_variants:
        matches = [col for col in df.columns if variant.lower() in col.lower()]
        if matches:
            column_mapping['credit_utilization'] = matches[0]
            break
    
    # Common variations for payment history
    payment_variants = ['payment_history', 'payment_score', 'payment_performance', 'pay_history', 'payment_rate']
    for variant in payment_variants:
        matches = [col for col in df.columns if variant.lower() in col.lower()]
        if matches:
            column_mapping['payment_history'] = matches[0]
            break
    
    # Common variations for loan amount
    loan_variants = ['loan_amount', 'amount', 'loan_value', 'principal', 'loan_size', 'request_amount']
    for variant in loan_variants:
        matches = [col for col in df.columns if variant.lower() in col.lower()]
        if matches:
            column_mapping['loan_amount'] = matches[0]
            break
    
    return column_mapping

def preprocess_uploaded_data(df):
    """Preprocess uploaded data to standardize column names and create missing features"""
    original_df = df.copy()
    
    # Detect column mappings
    column_mapping = detect_and_map_columns(df)
    
    # Rename columns based on mapping
    df_processed = df.rename(columns={v: k for k, v in column_mapping.items()})
    
    # Auto-detect numeric columns for features if standard columns are missing
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create synthetic credit_score if missing
    if 'credit_score' not in df_processed.columns:
        if len(numeric_cols) >= 1:
            # Use first numeric column as proxy for credit score
            score_col = numeric_cols[0]
            # Normalize to 300-850 range
            min_val, max_val = df[score_col].min(), df[score_col].max()
            if max_val > min_val:
                df_processed['credit_score'] = 300 + (df[score_col] - min_val) / (max_val - min_val) * 550
            else:
                df_processed['credit_score'] = np.random.normal(700, 80, len(df))
            st.info(f"📊 Created credit_score from column: {score_col}")
        else:
            # Generate synthetic credit scores
            df_processed['credit_score'] = np.random.normal(700, 80, len(df))
            st.info("📊 Generated synthetic credit scores")
    
    # Ensure credit_score is in valid range
    df_processed['credit_score'] = np.clip(df_processed['credit_score'], 300, 850)
    
    # Create synthetic income if missing
    if 'income' not in df_processed.columns:
        if len(numeric_cols) >= 2:
            # Use second numeric column as proxy for income
            income_col = [col for col in numeric_cols if col != column_mapping.get('credit_score', '')][0]
            # Ensure positive values and reasonable scale
            income_values = abs(df[income_col])
            if income_values.max() < 1000:  # Likely needs scaling
                df_processed['income'] = income_values * 1000
            else:
                df_processed['income'] = income_values
            st.info(f"💰 Created income from column: {income_col}")
        else:
            # Generate synthetic income based on credit score
            base_income = 30000 + (df_processed['credit_score'] - 300) / 550 * 100000
            df_processed['income'] = base_income + np.random.normal(0, 15000, len(df))
            df_processed['income'] = np.maximum(df_processed['income'], 15000)  # Minimum income
            st.info("💰 Generated synthetic income based on credit scores")
    
    # Create synthetic debt_to_income if missing
    if 'debt_to_income' not in df_processed.columns:
        if len(numeric_cols) >= 3:
            # Use third numeric column as proxy
            dti_col = [col for col in numeric_cols if col not in [
                column_mapping.get('credit_score', ''), 
                column_mapping.get('income', '')
            ]][0]
            # Normalize to 0-1 range
            dti_values = abs(df[dti_col])
            df_processed['debt_to_income'] = dti_values / dti_values.max() if dti_values.max() > 0 else 0.3
            st.info(f"📈 Created debt_to_income from column: {dti_col}")
        else:
            # Generate synthetic debt-to-income ratio based on credit score
            base_dti = 0.5 - (df_processed['credit_score'] - 300) / 550 * 0.3
            df_processed['debt_to_income'] = np.clip(base_dti + np.random.normal(0, 0.1, len(df)), 0.05, 0.95)
            st.info("📈 Generated synthetic debt-to-income ratios")
    
    # Create additional features from available data
    if 'age' not in df_processed.columns:
        df_processed['age'] = np.clip(np.random.normal(45, 15, len(df)), 18, 80)
        st.info("👥 Generated synthetic age data")
    
    if 'credit_utilization' not in df_processed.columns:
        # Base utilization on credit score (inverse relationship)
        base_util = 0.6 - (df_processed['credit_score'] - 300) / 550 * 0.4
        df_processed['credit_utilization'] = np.clip(base_util + np.random.normal(0, 0.1, len(df)), 0.01, 0.99)
        st.info("💳 Generated credit utilization based on credit scores")
    
    if 'payment_history' not in df_processed.columns:
        # Base payment history on credit score
        base_payment = 0.6 + (df_processed['credit_score'] - 300) / 550 * 0.35
        df_processed['payment_history'] = np.clip(base_payment + np.random.normal(0, 0.05, len(df)), 0.5, 1.0)
        st.info("📅 Generated payment history based on credit scores")
    
    if 'loan_amount' not in df_processed.columns:
        # Base loan amount on income
        df_processed['loan_amount'] = df_processed['income'] * np.random.uniform(0.1, 0.5, len(df))
        st.info("💵 Generated loan amounts based on income")
    
    # Create risk categories
    df_processed['risk_category'] = pd.cut(df_processed['credit_score'], 
                                         bins=[0, 580, 670, 740, 850], 
                                         labels=['High Risk', 'Medium Risk', 'Low Risk', 'Excellent'])
    
    # Create client IDs if missing
    if 'client_id' not in df_processed.columns:
        df_processed['client_id'] = [f"CLIENT_{i:06d}" for i in range(len(df_processed))]
    
    # Add employment length for enhanced features
    df_processed['employment_length'] = np.random.exponential(5, len(df))
    df_processed['num_accounts'] = np.random.poisson(8, len(df))
    df_processed['recent_inquiries'] = np.random.poisson(2, len(df))
    df_processed['delinquencies_2yrs'] = np.random.poisson(0.5, len(df))
    df_processed['total_credit_limit'] = df_processed['income'] * np.random.uniform(0.5, 2.0, len(df))
    df_processed['monthly_payment'] = df_processed['loan_amount'] / np.random.uniform(12, 60, len(df))
    df_processed['debt_consolidation'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    
    # Show mapping summary
    if column_mapping:
        st.success("✅ Column Mapping Detected:")
        for standard_name, original_name in column_mapping.items():
            st.write(f"- {standard_name} ← {original_name}")
    
    st.info(f"📊 Processed data: {len(df_processed)} records with {len(df_processed.columns)} features")
    
    return df_processed, column_mapping

def generate_enhanced_sample_data(n_samples=1000):
    """Generate enhanced sample credit data with more features"""
    np.random.seed(42)
    
    # Basic features
    data = {
        'client_id': [f"CLIENT_{i:06d}" for i in range(n_samples)],
        'credit_score': np.clip(np.random.normal(700, 80, n_samples), 300, 850),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'debt_to_income': np.random.beta(2, 5, n_samples),
        'payment_history': np.random.uniform(0.6, 1.0, n_samples),
        'credit_utilization': np.random.beta(2, 8, n_samples),
        'age': np.clip(np.random.normal(45, 15, n_samples), 18, 80),
        'loan_amount': np.random.lognormal(10, 0.8, n_samples),
        
        # Additional features
        'employment_length': np.random.exponential(5, n_samples),
        'num_accounts': np.random.poisson(8, n_samples),
        'recent_inquiries': np.random.poisson(2, n_samples),
        'delinquencies_2yrs': np.random.poisson(0.5, n_samples),
        'total_credit_limit': np.random.lognormal(12, 0.8, n_samples),
        'monthly_payment': np.random.lognormal(8, 0.6, n_samples),
        'debt_consolidation': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, p=[0.4, 0.3, 0.3]),
        'verification_status': np.random.choice(['Verified', 'Not Verified', 'Source Verified'], 
                                              n_samples, p=[0.4, 0.3, 0.3]),
        'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 
                                   'major_purchase', 'small_business'], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Create risk categories
    df['risk_category'] = pd.cut(df['credit_score'], 
                                bins=[0, 580, 670, 740, 850], 
                                labels=['High Risk', 'Medium Risk', 'Low Risk', 'Excellent'])
    
    # Add some realistic correlations
    df.loc[df['credit_score'] < 600, 'debt_to_income'] *= 1.4
    df.loc[df['credit_score'] > 750, 'payment_history'] *= 1.1
    df.loc[df['debt_consolidation'] == 1, 'debt_to_income'] *= 1.2
    df['credit_age'] = df['employment_length'] * np.random.uniform(0.5, 2.0, n_samples)
    
    return df

def create_3d_portfolio_visualization(df):
    """Create 3D portfolio visualization"""
    sample_df = df.sample(min(500, len(df)))
    
    fig = go.Figure(data=[go.Scatter3d(
        x=sample_df['income'],
        y=sample_df['debt_to_income'],
        z=sample_df['credit_score'],
        mode='markers',
        marker=dict(
            size=6,
            color=sample_df['credit_score'],
            colorscale='Viridis',
            opacity=0.7,
            colorbar=dict(title="Credit Score", x=1.1),
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        text=sample_df.apply(lambda row: 
            f"Client: {row['client_id']}<br>" +
            f"Score: {row['credit_score']:.0f}<br>" +
            f"Income: ${row['income']:,.0f}<br>" +
            f"D/I Ratio: {row['debt_to_income']:.2%}<br>" +
            f"Risk: {row['risk_category']}", axis=1),
        hovertemplate='%{text}<extra></extra>',
        name="Credit Portfolio"
    )])
    
    fig.update_layout(
        title="3D Credit Portfolio Analysis",
        scene=dict(
            xaxis_title="Annual Income ($)",
            yaxis_title="Debt-to-Income Ratio",
            zaxis_title="Credit Score",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor='white'
        ),
        height=600,
        margin=dict(l=0, r=50, t=50, b=0)
    )
    
    return fig

def create_risk_heatmap(df):
    """Create risk heatmap by income and age groups"""
    # Create income and age bins
    df_temp = df.copy()
    df_temp['income_group'] = pd.cut(df_temp['income'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    df_temp['age_group'] = pd.cut(df_temp['age'], bins=5, labels=['Young', 'Young Adult', 'Middle Age', 'Mature', 'Senior'])
    
    # Create pivot table for heatmap
    risk_matrix = pd.crosstab(df_temp['age_group'], df_temp['income_group'], 
                             values=(df_temp['risk_category'] == 'High Risk').astype(int), 
                             aggfunc='mean') * 100
    
    fig = px.imshow(
        risk_matrix,
        labels=dict(x="Income Group", y="Age Group", color="High Risk %"),
        title="Risk Heatmap by Age and Income Groups",
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=400)
    return fig

def create_advanced_correlation_network(df):
    """Create network-style correlation visualization"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    # Create network graph
    fig = go.Figure()
    
    # Add nodes
    for i, col in enumerate(numeric_cols):
        fig.add_trace(go.Scatter(
            x=[i], y=[0],
            mode='markers+text',
            marker=dict(size=20, color='lightblue'),
            text=[col],
            textposition='middle center',
            name=col,
            showlegend=False
        ))
    
    # Add edges for strong correlations
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j and abs(corr_matrix.loc[col1, col2]) > 0.5:
                fig.add_trace(go.Scatter(
                    x=[i, j], y=[0, 0],
                    mode='lines',
                    line=dict(
                        width=abs(corr_matrix.loc[col1, col2]) * 10,
                        color='red' if corr_matrix.loc[col1, col2] > 0 else 'blue'
                    ),
                    showlegend=False,
                    hovertemplate=f'{col1} - {col2}<br>Correlation: {corr_matrix.loc[col1, col2]:.3f}<extra></extra>'
                ))
    
    fig.update_layout(
        title="Feature Correlation Network (|r| > 0.5)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400
    )
    
    return fig

def create_pca_visualization(df):
    """Create PCA visualization"""
    numeric_cols = ['credit_score', 'income', 'debt_to_income', 'payment_history', 
                   'credit_utilization', 'age', 'loan_amount', 'employment_length']
    
    X = df[numeric_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X_scaled)
    
    st.session_state.pca_data = {
        'components': pca_result,
        'explained_variance': pca.explained_variance_ratio_,
        'feature_names': numeric_cols
    }
    
    fig = go.Figure(data=[go.Scatter3d(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        z=pca_result[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=df['credit_score'],
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title="Credit Score")
        ),
        text=[f"Client: {client}" for client in df['client_id']],
        hovertemplate='%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f"PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})",
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
        ),
        height=500
    )
    
    return fig

def create_clustering_analysis(df):
    """Perform clustering analysis"""
    numeric_cols = ['credit_score', 'income', 'debt_to_income', 'payment_history', 'credit_utilization']
    X = df[numeric_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    st.session_state.cluster_data = df_clustered
    
    fig = px.scatter_3d(
        df_clustered,
        x='income',
        y='credit_score',
        z='debt_to_income',
        color='cluster',
        title="Customer Segmentation (K-Means Clustering)",
        hover_data=['risk_category'],
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_layout(height=500)
    return fig

def create_model_comparison_dashboard(df):
    """Compare multiple ML models"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Prepare data
    feature_columns = ['credit_score', 'income', 'debt_to_income', 'payment_history', 'credit_utilization']
    X = df[feature_columns].fillna(0)
    
    le = LabelEncoder()
    y = le.fit_transform(df['risk_category'].fillna('Medium Risk'))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    st.session_state.models_comparison = results
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.classes = le.classes_
    
    return results

def create_roc_curves(models_results):
    """Create ROC curves for model comparison"""
    fig = go.Figure()
    
    for name, results in models_results.items():
        y_pred_proba = results['y_pred_proba']
        
        # For multi-class, use one-vs-rest
        for i, class_name in enumerate(st.session_state.classes):
            fpr, tpr, _ = roc_curve((st.session_state.y_test == i).astype(int), 
                                  y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{name} - {class_name} (AUC = {roc_auc:.3f})',
                line=dict(width=2)
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='black')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    
    return fig

def render_shap_analysis():
    """Enhanced SHAP analysis section"""
    if not SHAP_AVAILABLE:
        st.warning("🔧 SHAP is not installed. To enable explainability features:")
        st.code("pip install shap", language="bash")
        return
    
    if st.session_state.models_comparison is None:
        st.info("Please train models first to see SHAP explanations.")
        return
    
    st.header("🧠 Advanced Model Explainability (SHAP)")
    
    # Model selector
    model_name = st.selectbox("Select Model for SHAP Analysis", 
                             list(st.session_state.models_comparison.keys()),
                             key="shap_model_selector")
    
    model = st.session_state.models_comparison[model_name]['model']
    X_test = st.session_state.X_test
    
    # Generate SHAP values
    with st.spinner("Generating SHAP explanations..."):
        explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') or hasattr(model, 'estimators_') else shap.Explainer(model)
        shap_values = explainer.shap_values(X_test)
    
    # SHAP Summary Plot
    st.subheader("📊 Feature Importance Summary")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_test, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    
    # Individual explanation
    st.subheader("🔍 Individual Client Explanation")
    
    client_idx = st.slider("Select Client", 0, len(X_test)-1, 0, key="shap_client_slider")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Client Data:**")
        client_data = X_test.iloc[client_idx]
        st.dataframe(client_data.to_frame().T)
    
    with col2:
        pred = model.predict([client_data])[0]
        pred_proba = model.predict_proba([client_data])[0]
        predicted_class = st.session_state.classes[pred]
        confidence = pred_proba.max()
        
        st.write("**Prediction:**")
        st.metric("Risk Category", str(predicted_class))
        st.metric("Confidence", f"{confidence:.1%}")
    
    # Force plot
    st.subheader("💡 Feature Contribution Analysis")
    try:
        if isinstance(shap_values, list):
            class_idx = st.selectbox("Select Class", range(len(st.session_state.classes)), 
                                   format_func=lambda x: st.session_state.classes[x],
                                   key="shap_class_selector")
            shap_vals = shap_values[class_idx][client_idx]
            expected_val = explainer.expected_value[class_idx]
        else:
            shap_vals = shap_values[client_idx]
            expected_val = explainer.expected_value
        
        # Create feature contribution plot
        explanation_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Value': client_data.values,
            'SHAP_Value': shap_vals,
            'Contribution': ['Positive' if val > 0 else 'Negative' for val in shap_vals]
        }).sort_values('SHAP_Value', key=abs, ascending=False)
        
        fig = px.bar(
            explanation_df,
            x='SHAP_Value',
            y='Feature',
            color='Contribution',
            title=f"SHAP Values for {predicted_class} Prediction",
            color_discrete_map={'Positive': 'green', 'Negative': 'red'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating SHAP plot: {str(e)}")

def main():
    """Main application with enhanced features"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 CredTech Advanced Intelligence Platform</h1>
        <p>Comprehensive Credit Analytics with 3D Visualizations & AI Explainability</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ 3D Visualizations Enabled")
    with col2:
        if SHAP_AVAILABLE:
            st.success("✅ SHAP Explainability Available")
        else:
            st.warning("⚠️ SHAP Not Available")
    with col3:
        st.info(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Data source selection
    data_source = st.sidebar.radio("Data Source", ["Generate Sample", "Upload File"], key="data_source_radio")
    
    if data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'], key="file_uploader")
        if uploaded_file:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.sidebar.info(f"📁 Raw data: {len(raw_df)} records, {len(raw_df.columns)} columns")
                
                # Process and validate the uploaded data
                with st.sidebar:
                    with st.spinner("Processing uploaded data..."):
                        df, success, message = validate_and_process_uploaded_data(raw_df)
                
                if success:
                    st.sidebar.success(f"✅ {message}")
                    st.sidebar.success(f"📊 Processed: {len(df)} records")
                    
                    # Show column mapping info
                    with st.sidebar.expander("📋 Data Processing Details"):
                        st.write("**Original columns:**")
                        st.write(list(raw_df.columns))
                        st.write("**Processed columns:**")
                        st.write(list(df.columns))
                        st.write("**Missing data filled:** ✅")
                        st.write("**Risk categories created:** ✅")
                else:
                    st.sidebar.error(f"❌ {message}")
                    df = generate_enhanced_sample_data(500)  # Fallback to sample data
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
                df = generate_enhanced_sample_data(500)  # Fallback to sample data
        else:
            df = generate_enhanced_sample_data(500)
    else:
        sample_size = st.sidebar.slider("Sample Size", 100, 2000, 1000, key="sample_size_slider")
        df = generate_enhanced_sample_data(sample_size)
        st.sidebar.info(f"📊 Generated {len(df)} records")
    
    # Advanced options
    st.sidebar.subheader("🔧 Advanced Options")
    show_3d = st.sidebar.checkbox("Enable 3D Visualizations", True, key="show_3d_checkbox")
    show_clustering = st.sidebar.checkbox("Enable Clustering Analysis", True, key="show_clustering_checkbox")
    show_pca = st.sidebar.checkbox("Enable PCA Analysis", True, key="show_pca_checkbox")
    
    # Main dashboard
    st.header("📊 Enhanced Portfolio Analytics")
    
    # Validate that we have the required columns
    required_columns = ['credit_score', 'income', 'debt_to_income', 'risk_category']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        st.info("📝 Please ensure your CSV file contains credit scoring data or use the sample data generator.")
        return
    
    # Key metrics with enhanced styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    try:
        metrics = [
            ("Avg Credit Score", f"{df['credit_score'].mean():.0f}", "📈"),
            ("High Risk %", f"{(df['risk_category'] == 'High Risk').mean():.1%}", "⚠️"),
            ("Avg Income", f"${df['income'].mean():,.0f}", "💰"),
            ("Total Clients", f"{len(df):,}", "👥"),
            ("Avg Utilization", f"{df['credit_utilization'].mean():.1%}" if 'credit_utilization' in df.columns else "N/A", "📊")
        ]
        
        for i, (col, (label, value, icon)) in enumerate(zip([col1, col2, col3, col4, col5], metrics)):
            with col:
                st.metric(f"{icon} {label}", value)
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return
    
    # Advanced visualizations
    st.header("🎨 Advanced Visualizations")
    
    if show_3d:
        st.subheader("🌐 3D Portfolio Analysis")
        fig_3d = create_3d_portfolio_visualization(df)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.info("💡 **3D Controls:** Drag to rotate, scroll to zoom, double-click to reset view")
    
    # Traditional charts in enhanced layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk heatmap
        fig_heatmap = create_risk_heatmap(df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        if show_pca:
            # PCA visualization
            fig_pca = create_pca_visualization(df)
            st.plotly_chart(fig_pca, use_container_width=True)
    
    with col2:
        # Correlation network
        fig_network = create_advanced_correlation_network(df)
        st.plotly_chart(fig_network, use_container_width=True)
        
        if show_clustering:
            # Clustering analysis
            fig_cluster = create_clustering_analysis(df)
            st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Machine Learning Section
    st.header("🤖 Advanced Machine Learning Analysis")
    
    if st.button("🚀 Train Multiple Models", type="primary", key="train_models_btn"):
        with st.spinner("Training multiple models..."):
            results = create_model_comparison_dashboard(df)
        st.success("✅ All models trained successfully!")
    
    # Model comparison results
    if st.session_state.models_comparison:
        st.subheader("📈 Model Performance Comparison")
        
        # Performance metrics
        performance_data = []
        for name, results in st.session_state.models_comparison.items():
            performance_data.append({
                'Model': name,
                'Accuracy': f"{results['accuracy']:.1%}",
                'Accuracy_Value': results['accuracy']
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        fig_perf = px.bar(
            performance_df,
            x='Model',
            y='Accuracy_Value',
            title="Model Accuracy Comparison",
            text='Accuracy'
        )
        fig_perf.update_traces(textposition='outside')
        fig_perf.update_layout(yaxis_title="Accuracy", height=400)
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # ROC Curves
        st.subheader("📊 ROC Curves Analysis")
        fig_roc = create_roc_curves(st.session_state.models_comparison)
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # SHAP Analysis
    render_shap_analysis()
    
    # Clustering insights
    if show_clustering and st.session_state.cluster_data is not None:
        st.header("🎯 Customer Segmentation Insights")
        
        cluster_summary = st.session_state.cluster_data.groupby('cluster').agg({
            'credit_score': 'mean',
            'income': 'mean',
            'debt_to_income': 'mean',
            'risk_category': lambda x: (x == 'High Risk').mean()
        }).round(2)
        
        cluster_summary.columns = ['Avg Credit Score', 'Avg Income', 'Avg D/I Ratio', 'High Risk %']
        st.dataframe(cluster_summary, use_container_width=True)
    
    # Data explorer
    with st.expander("🔍 Advanced Data Explorer"):
        st.subheader("Filter and Explore Data")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score_range = st.slider("Credit Score Range", 300, 850, (300, 850), key="score_filter")
        with col2:
            income_range = st.slider("Income Range (K)", 0, int(df['income'].max()/1000), 
                                   (0, int(df['income'].max()/1000)), key="income_filter")
        with col3:
            risk_filter = st.multiselect("Risk Categories", df['risk_category'].unique().tolist(),
                                       default=df['risk_category'].unique().tolist(), key="risk_filter")
        
        # Apply filters
        filtered_df = df[
            (df['credit_score'] >= score_range[0]) & 
            (df['credit_score'] <= score_range[1]) &
            (df['income'] >= income_range[0] * 1000) & 
            (df['income'] <= income_range[1] * 1000) &
            (df['risk_category'].isin(risk_filter))
        ]
        
        st.info(f"Filtered data: {len(filtered_df)} records out of {len(df)}")
        st.dataframe(filtered_df.head(100), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 **CredTech Advanced Intelligence Platform** - Powered by AI & Machine Learning")

if __name__ == "__main__":
    main()
