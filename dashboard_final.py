"""
CredTech Intelligence Platform - Simple Dashboard (SHAP-Ready)
============================================================

A streamlined credit dashboard with optional SHAP explainability.
SHAP will be loaded dynamically if available.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
import streamlit.components.v1 as components

# Try to import SHAP, but make it optional
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP is available")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP is not available. Install with: pip install shap")

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="CredTech Credit Dashboard",
    page_icon="🏛️",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'classes' not in st.session_state:
    st.session_state.classes = None
if 'shap_explainer' not in st.session_state:
    st.session_state.shap_explainer = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None

def generate_sample_data(n_samples=500):
    """Generate sample credit data for demonstration"""
    np.random.seed(42)
    
    data = {
        'client_id': [f"CLIENT_{i:06d}" for i in range(n_samples)],
        'credit_score': np.clip(np.random.normal(700, 80, n_samples), 300, 850),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'debt_to_income': np.random.beta(2, 5, n_samples),
        'payment_history': np.random.uniform(0.6, 1.0, n_samples),
        'credit_utilization': np.random.beta(2, 8, n_samples),
        'age': np.clip(np.random.normal(45, 15, n_samples), 18, 80),
        'loan_amount': np.random.lognormal(10, 0.8, n_samples)
    }
    
    df = pd.DataFrame(data)
    df['risk_category'] = pd.cut(df['credit_score'], 
                                bins=[0, 580, 670, 740, 850], 
                                labels=['High Risk', 'Medium Risk', 'Low Risk', 'Excellent'])
    
    return df

def validate_uploaded_data(df):
    """Validate uploaded data format and structure"""
    required_columns = ['credit_score', 'income', 'debt_to_income']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    if df['credit_score'].dtype not in ['int64', 'float64']:
        return False, "Credit score must be numeric"
    
    if (df['credit_score'] < 300).any() or (df['credit_score'] > 850).any():
        return False, "Credit scores must be between 300 and 850"
    
    return True, "Data validation passed"

def st_shap(plot, height=None):
    """Display SHAP plot in Streamlit"""
    if SHAP_AVAILABLE:
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    else:
        st.error("SHAP is not available. Please install it with: pip install shap")

def explain_model_predictions(model, X_test, feature_names):
    """Generate SHAP explanations for the model"""
    if not SHAP_AVAILABLE:
        st.error("SHAP is not installed. Please install it with: `pip install shap`")
        return None, None
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        return explainer, shap_values
    except Exception as e:
        st.error(f"Error generating SHAP explanations: {str(e)}")
        return None, None

def train_model(df):
    """Train a Random Forest model on the credit data"""
    # Prepare features
    feature_columns = ['credit_score', 'income', 'debt_to_income']
    
    # Add optional columns if they exist
    optional_columns = ['payment_history', 'credit_utilization', 'age', 'loan_amount']
    for col in optional_columns:
        if col in df.columns:
            feature_columns.append(col)
    
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
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    
    return model, X_test, y_test, report, le.classes_, accuracy

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
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_score_histogram(df):
    """Create credit score distribution histogram"""
    fig = px.histogram(
        df,
        x='credit_score',
        nbins=30,
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
        height=400
    )
    
    return fig

def create_income_vs_score_scatter(df):
    """Create income vs credit score scatter plot"""
    sample_df = df.sample(min(500, len(df)))
    
    fig = px.scatter(
        sample_df,
        x='income',
        y='credit_score',
        color='risk_category',
        title="Income vs Credit Score Analysis",
        hover_data=['age'] if 'age' in df.columns else None,
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
        height=400
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap of numerical features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'client_id' in numeric_cols:
        numeric_cols.remove('client_id')
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        x=numeric_cols,
        y=numeric_cols,
        color_continuous_scale='RdBu',
        aspect='auto',
        title="Feature Correlation Heatmap"
    )
    
    fig.update_layout(height=400)
    return fig

def render_shap_section():
    """Render SHAP explainability section"""
    if not SHAP_AVAILABLE:
        st.warning("🔧 SHAP is not installed. To enable model explainability features, run:")
        st.code("pip install shap", language="bash")
        
        with st.expander("💡 What is SHAP?"):
            st.write("""
            **SHAP (SHapley Additive exPlanations)** is a unified approach to explain machine learning model predictions.
            
            With SHAP installed, you would see:
            - 📊 Feature importance rankings
            - 🔍 Individual prediction explanations  
            - 💡 Feature contribution analysis
            - 🌊 Waterfall charts showing feature impacts
            """)
        return
    
    if st.session_state.shap_values is not None and st.session_state.shap_explainer is not None:
        st.header("🧠 Model Explainability (SHAP)")
        
        # Feature importance summary
        st.subheader("📊 Feature Impact Summary")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check if shap_values is a list (multi-class) or array (binary)
        if isinstance(st.session_state.shap_values, list):
            # Multi-class case - use the first class for summary
            shap.summary_plot(st.session_state.shap_values[0], st.session_state.X_test, 
                            plot_type="bar", show=False)
        else:
            # Binary case
            shap.summary_plot(st.session_state.shap_values, st.session_state.X_test, 
                            plot_type="bar", show=False)
        
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        
        # Individual prediction explanation
        st.subheader("🔍 Individual Prediction Explanation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            client_idx = st.slider(
                "Select Client from Test Set", 
                0, len(st.session_state.X_test)-1, 0, 
                key="shap_client_selector_slider"
            )
            
            st.write("**Client Data:**")
            client_data = st.session_state.X_test.iloc[client_idx]
            st.dataframe(client_data.to_frame().T)
        
        with col2:
            # Prediction for this client
            if st.session_state.model is not None:
                pred = st.session_state.model.predict([client_data])[0]
                pred_proba = st.session_state.model.predict_proba([client_data])[0]
                predicted_class = st.session_state.classes[pred]
                confidence = pred_proba.max()
                
                st.write("**Model Prediction:**")
                st.metric("Predicted Risk Category", str(predicted_class))
                st.metric("Confidence", f"{confidence:.1%}")
        
        # SHAP Force Plot
        st.subheader("💡 Feature Contribution Analysis")
        
        if isinstance(st.session_state.shap_values, list):
            # Multi-class case
            class_idx = st.selectbox(
                "Select Risk Category to Explain", 
                range(len(st.session_state.classes)), 
                format_func=lambda x: st.session_state.classes[x],
                key="shap_class_selector"
            )
            
            try:
                force_plot = shap.force_plot(
                    st.session_state.shap_explainer.expected_value[class_idx],
                    st.session_state.shap_values[class_idx][client_idx],
                    st.session_state.X_test.iloc[client_idx],
                    matplotlib=True,
                    show=False
                )
                st.pyplot(force_plot, clear_figure=True)
            except Exception as e:
                st.error(f"Error generating force plot: {str(e)}")
                
        else:
            # Binary case
            try:
                force_plot = shap.force_plot(
                    st.session_state.shap_explainer.expected_value,
                    st.session_state.shap_values[client_idx],
                    st.session_state.X_test.iloc[client_idx],
                    matplotlib=True,
                    show=False
                )
                st.pyplot(force_plot, clear_figure=True)
            except Exception as e:
                st.error(f"Error generating force plot: {str(e)}")
        
        # Feature importance table as alternative
        st.subheader("📋 Feature Contribution Details")
        if isinstance(st.session_state.shap_values, list):
            shap_vals = st.session_state.shap_values[0][client_idx]
        else:
            shap_vals = st.session_state.shap_values[client_idx]
            
        explanation_df = pd.DataFrame({
            'Feature': st.session_state.X_test.columns,
            'Feature_Value': client_data.values,
            'SHAP_Value': shap_vals,
            'Impact': ['Positive' if val > 0 else 'Negative' for val in shap_vals]
        }).sort_values('SHAP_Value', key=abs, ascending=False)
        
        st.dataframe(explanation_df, use_container_width=True)
    
    elif st.session_state.model is not None:
        st.info("🧠 SHAP explanations will be generated when you train the model.")

def main():
    """Main application"""
    # Header
    st.title("🏛️ CredTech Intelligence Platform")
    st.markdown("**Comprehensive Credit Scoring & Analysis Dashboard**")
    
    # Show SHAP status
    if SHAP_AVAILABLE:
        st.success("✅ SHAP model explainability is enabled")
    else:
        st.warning("⚠️ SHAP not available - install with `pip install shap` for explainability features")
    
    # Sidebar
    st.sidebar.header("📋 Data Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Credit Data (CSV)",
        type=['csv'],
        help="Upload a CSV file with credit data. Required columns: credit_score, income, debt_to_income",
        key="file_uploader_widget"
    )
    
    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            is_valid, validation_message = validate_uploaded_data(df)
            if is_valid:
                st.sidebar.success(f"✅ {validation_message}")
                st.sidebar.info(f"📊 {len(df)} records loaded")
            else:
                st.sidebar.error(f"❌ {validation_message}")
                return
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
            return
    else:
        # Generate sample data
        sample_size = st.sidebar.slider("Sample Data Size", 100, 2000, 500, 100, key="sample_size_main_slider")
        if st.sidebar.button("🔄 Generate New Sample", key="generate_sample_main_btn"):
            df = generate_sample_data(sample_size)
            st.sidebar.success(f"✅ Generated {sample_size} sample records")
        else:
            df = generate_sample_data(sample_size)
    
    # Main dashboard
    st.header("📊 Portfolio Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = df['credit_score'].mean()
        st.metric("Average Credit Score", f"{avg_score:.0f}")
    
    with col2:
        high_risk_pct = (df['risk_category'] == 'High Risk').mean() * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%")
    
    with col3:
        avg_income = df['income'].mean()
        st.metric("Average Income", f"${avg_income:,.0f}")
    
    with col4:
        if 'credit_utilization' in df.columns:
            avg_utilization = df['credit_utilization'].mean() * 100
            st.metric("Avg Utilization", f"{avg_utilization:.1f}%")
        else:
            st.metric("Total Records", f"{len(df):,}")
    
    # Visualizations
    st.header("📈 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        fig_risk = create_risk_distribution_chart(df)
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Income vs Score scatter
        fig_scatter = create_income_vs_score_scatter(df)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Score histogram
        fig_hist = create_score_histogram(df)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Correlation heatmap
        fig_corr = create_correlation_heatmap(df)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Model Training Section
    st.header("🤖 Machine Learning Model")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🚀 Train Model", type="primary", key="train_model_main_btn"):
            with st.spinner("Training Random Forest model..."):
                model, X_test, y_test, report, classes, accuracy = train_model(df)
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.report = report
                st.session_state.classes = classes
                
                # Generate SHAP explanations if available
                if SHAP_AVAILABLE:
                    with st.spinner("Generating SHAP explanations..."):
                        explainer, shap_values = explain_model_predictions(model, X_test, X_test.columns)
                        st.session_state.shap_explainer = explainer
                        st.session_state.shap_values = shap_values
                    
            st.success(f"✅ Model trained! Accuracy: {accuracy:.1%}")
            if SHAP_AVAILABLE and st.session_state.shap_values is not None:
                st.success("✅ SHAP explanations generated!")
    
    with col2:
        if st.session_state.model is not None:
            st.success("🎯 Model is trained and ready")
            st.info(f"Features used: {list(st.session_state.X_test.columns)}")
            if SHAP_AVAILABLE:
                st.info("🧠 SHAP explanations are available below")
        else:
            st.info("👆 Click 'Train Model' to train a Random Forest classifier")
    
    # Model Results
    if st.session_state.report is not None:
        st.subheader("📋 Model Performance")
        
        # Classification report
        report_df = pd.DataFrame(st.session_state.report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Feature importance
        if hasattr(st.session_state.model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': st.session_state.X_test.columns,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Random Forest Feature Importance"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # SHAP Explainability Section
    render_shap_section()
    
    # Data preview
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)

if __name__ == "__main__":
    main()
