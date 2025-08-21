# Entry point for Streamlit dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import logging

# Configure page
st.set_page_config(
    page_title="Credit Intelligence Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_css():
    """Load custom CSS styling."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def create_sample_data():
    """Create sample data for demonstration."""
    dates = pd.date_range(start='2025-01-01', end='2025-08-22', freq='D')
    
    # Sample credit scores over time
    credit_scores = 650 + np.random.normal(0, 20, len(dates))
    credit_scores = np.clip(credit_scores, 300, 850)
    
    # Sample application volumes
    applications = np.random.poisson(100, len(dates))
    
    return pd.DataFrame({
        'date': dates,
        'credit_score': credit_scores,
        'applications': applications,
        'risk_level': pd.cut(credit_scores, bins=[0, 550, 700, 850], 
                           labels=['High', 'Medium', 'Low'])
    })

def main():
    """Main dashboard application."""
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Intelligence Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        max_value=datetime.now()
    )
    
    # Model selector
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["LightGBM", "XGBoost", "Ensemble", "Neural Network"]
    )
    
    # Real-time toggle
    real_time = st.sidebar.checkbox("Enable Real-time Updates", value=False)
    
    if real_time:
        st.sidebar.info("🔄 Real-time mode enabled")
        # Auto-refresh every 30 seconds in real implementation
        st.sidebar.write("Last updated:", datetime.now().strftime("%H:%M:%S"))
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Key metrics
    with col1:
        st.metric(
            label="📊 Average Credit Score",
            value="685",
            delta="5.2",
            help="Average credit score across all applications"
        )
    
    with col2:
        st.metric(
            label="📈 Applications Today",
            value="247",
            delta="12",
            help="Number of credit applications processed today"
        )
    
    with col3:
        st.metric(
            label="⚠️ High Risk Applications",
            value="18%",
            delta="-2.1%",
            delta_color="inverse",
            help="Percentage of applications flagged as high risk"
        )
    
    with col4:
        st.metric(
            label="🎯 Model Accuracy",
            value="94.2%",
            delta="0.8%",
            help="Current model prediction accuracy"
        )
    
    # Charts section
    st.markdown("---")
    
    # Load sample data
    df = create_sample_data()
    
    # Filter data based on date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['date'] >= pd.Timestamp(start_date)) & 
                (df['date'] <= pd.Timestamp(end_date))]
    
    # Credit score trend
    st.subheader("📈 Credit Score Trends")
    fig_trend = px.line(
        df, 
        x='date', 
        y='credit_score',
        title=f"Credit Score Trends - {model_type} Model",
        color_discrete_sequence=['#1f77b4']
    )
    fig_trend.update_layout(height=400)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Two column layout for additional charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🎯 Risk Distribution")
        
        # Risk level distribution
        risk_counts = df['risk_level'].value_counts()
        colors = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution",
            color=risk_counts.index,
            color_discrete_map=colors
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col_right:
        st.subheader("📊 Application Volume")
        
        # Daily application volumes
        fig_apps = px.bar(
            df.tail(30),  # Last 30 days
            x='date',
            y='applications',
            title="Daily Application Volumes",
            color_discrete_sequence=['#17a2b8']
        )
        fig_apps.update_layout(height=300)
        st.plotly_chart(fig_apps, use_container_width=True)
    
    # Recent applications table
    st.markdown("---")
    st.subheader("📋 Recent Applications")
    
    # Sample recent applications data
    recent_apps = pd.DataFrame({
        'Application ID': [f'APP-{1000+i}' for i in range(10)],
        'Score': np.random.randint(300, 850, 10),
        'Risk Level': np.random.choice(['Low', 'Medium', 'High'], 10),
        'Status': np.random.choice(['Approved', 'Pending', 'Declined'], 10),
        'Timestamp': [datetime.now() - timedelta(minutes=i*15) for i in range(10)]
    })
    
    # Style the dataframe
    def style_risk_level(val):
        if val == 'Low':
            return 'color: #28a745'
        elif val == 'Medium':
            return 'color: #ffc107'
        else:
            return 'color: #dc3545'
    
    styled_df = recent_apps.style.applymap(
        style_risk_level, 
        subset=['Risk Level']
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Credit Intelligence Platform** - Real-time credit risk assessment and monitoring",
        help="Powered by advanced ML models and real-time data processing"
    )

if __name__ == "__main__":
    main()
