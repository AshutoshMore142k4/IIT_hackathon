"""
Comprehensive configuration management system for the credit intelligence platform.

This module provides environment-based configuration loading with validation
for API keys and settings across development, staging, and production environments.
"""

import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class APIConfig:
    """API-related configuration settings."""
    sec_edgar_base_url: str = "https://data.sec.gov/submissions/"
    sec_edgar_rate_limit: int = 10  # requests per second
    sec_edgar_user_agent: str = "CreditIntelligence/1.0 (contact@creditintel.com)"
    
    news_api_key: Optional[str] = None
    news_api_endpoint: str = "https://newsapi.org/v2/"
    news_rate_limit: int = 1000  # requests per day
    
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 1000
    openai_temperature: float = 0.7
    
    database_dev_url: str = "sqlite:///./credit_intel_dev.db"
    database_staging_url: str = "postgresql://user:pass@localhost:5432/credit_intel_staging"
    database_prod_url: str = "postgresql://user:pass@prod-host:5432/credit_intel_prod"
    database_url: Optional[str] = None  # Will be set based on environment


@dataclass
class DataPipelineConfig:
    """Data pipeline configuration settings."""
    update_intervals: Dict[str, int] = field(default_factory=lambda: {
        "realtime": 30,  # minutes
        "batch_daily": 1440,  # minutes (24 hours)
        "batch_weekly": 10080,  # minutes (7 days)
        "news_feed": 15  # minutes
    })
    
    supported_tickers: List[str] = field(default_factory=lambda: [
        # Top 50 S&P 500 companies by market cap
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "TSLA", "META", "TSM", "AVGO",
        "BRK.B", "UNH", "LLY", "JNJ", "V", "XOM", "WMT", "JPM", "PG", "MA",
        "ORCL", "HD", "CVX", "ABBV", "KO", "PEP", "ADBE", "BAC", "ASML", "COST",
        "MRK", "NFLX", "CRM", "ACN", "AMD", "LIN", "TMO", "CSCO", "DHR", "ABT",
        "TXN", "DIS", "VZ", "NKE", "QCOM", "WFC", "IBM", "INTC", "AMGN", "PM"
    ])
    
    feature_engineering_params: Dict[str, int] = field(default_factory=lambda: {
        "lookback_days": 252,  # 1 trading year
        "short_ma_period": 20,
        "long_ma_period": 50,
        "volatility_window": 30,
        "rsi_period": 14,
        "sentiment_lookback": 7
    })
    
    data_retention_days: int = 365 * 3  # 3 years


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "xgboost": 0.4,
        "random_forest": 0.3,
        "logistic_regression": 0.3
    })
    
    explainability_settings: Dict[str, int] = field(default_factory=lambda: {
        "shap_sample_size": 1000,
        "explanation_depth": 10,
        "max_features_display": 20
    })
    
    uncertainty_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "confidence_interval": 0.95,
        "prediction_threshold": 0.5,
        "high_risk_threshold": 0.8,
        "low_risk_threshold": 0.2
    })
    
    model_retraining_frequency: int = 7  # days
    model_validation_split: float = 0.2
    cross_validation_folds: int = 5


@dataclass
class DashboardConfig:
    """Dashboard and UI configuration."""
    refresh_intervals: Dict[str, int] = field(default_factory=lambda: {
        "charts": 30,  # seconds
        "alerts": 60,  # seconds
        "portfolio_summary": 120,  # seconds
        "news_feed": 300  # seconds
    })
    
    ui_themes: List[str] = field(default_factory=lambda: ["light", "dark", "auto"])
    default_theme: str = "auto"
    
    mobile_breakpoints: Dict[str, int] = field(default_factory=lambda: {
        "xs": 480,
        "sm": 768,
        "md": 1024,
        "lg": 1200,
        "xl": 1400
    })
    
    max_chart_points: int = 1000
    session_timeout_minutes: int = 120


@dataclass
class DeploymentConfig:
    """Deployment and infrastructure configuration."""
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8501",  # Streamlit default
        "https://creditintel.com",
        "https://app.creditintel.com"
    ])
    
    rate_limiting: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "default": {"requests": 100, "window": 3600},  # 100 req/hour
        "predictions": {"requests": 50, "window": 3600},  # 50 req/hour
        "explanations": {"requests": 20, "window": 3600},  # 20 req/hour
        "data_upload": {"requests": 10, "window": 3600}  # 10 req/hour
    })
    
    logging_level: str = "INFO"
    log_file_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_file_backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self, environment: Optional[str] = None):
        """Initialize configuration based on environment."""
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        
        # Initialize configuration sections
        self.api = APIConfig()
        self.data_pipeline = DataPipelineConfig()
        self.model = ModelConfig()
        self.dashboard = DashboardConfig()
        self.deployment = DeploymentConfig()
        
        # Load environment-specific overrides
        self._load_environment_config()
        self._validate_required_keys()
        
    def _load_environment_config(self):
        """Load environment-specific configuration overrides."""
        # API Keys from environment variables
        self.api.news_api_key = os.getenv("NEWS_API_KEY")
        self.api.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Database URLs based on environment
        if self.environment == "production":
            self.api.database_url = os.getenv("DATABASE_URL", self.api.database_prod_url)
            self.deployment.logging_level = "WARNING"
            # Restrict CORS in production
            self.deployment.cors_origins = [
                origin for origin in self.deployment.cors_origins 
                if not origin.startswith("http://localhost")
            ]
        elif self.environment == "staging":
            self.api.database_url = os.getenv("DATABASE_URL", self.api.database_staging_url)
            self.deployment.logging_level = "INFO"
        else:  # development
            self.api.database_url = os.getenv("DATABASE_URL", self.api.database_dev_url)
            self.deployment.logging_level = "DEBUG"
            
        # Override settings from environment variables
        self._load_env_overrides()
        
    def _load_env_overrides(self):
        """Load specific overrides from environment variables."""
        # Model configuration overrides
        model_retraining_freq = os.getenv("MODEL_RETRAINING_FREQUENCY")
        if model_retraining_freq:
            self.model.model_retraining_frequency = int(model_retraining_freq)
            
        # Rate limiting overrides
        api_rate_limit = os.getenv("API_RATE_LIMIT")
        if api_rate_limit:
            rate_limit = int(api_rate_limit)
            for endpoint in self.deployment.rate_limiting:
                self.deployment.rate_limiting[endpoint]["requests"] = rate_limit
                
        # Dashboard refresh overrides
        dashboard_refresh = os.getenv("DASHBOARD_REFRESH_INTERVAL")
        if dashboard_refresh:
            refresh_interval = int(dashboard_refresh)
            self.dashboard.refresh_intervals["charts"] = refresh_interval
            
    def _validate_required_keys(self):
        """Validate that required API keys are present for the current environment."""
        required_keys = []
        
        if self.environment == "production":
            required_keys = ["NEWS_API_KEY", "OPENAI_API_KEY"]
        elif self.environment == "staging":
            required_keys = ["OPENAI_API_KEY"]
            
        missing_keys = []
        for key in required_keys:
            if not os.getenv(key):
                missing_keys.append(key)
                
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
            
    def get_database_url(self) -> str:
        """Get the appropriate database URL for the current environment."""
        return getattr(self.api, 'database_url', self.api.database_dev_url)
        
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
        
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
        
    def get_logging_config(self) -> Dict:
        """Get logging configuration dictionary."""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': self.deployment.log_format
                },
            },
            'handlers': {
                'default': {
                    'level': self.deployment.logging_level,
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                },
                'file': {
                    'level': self.deployment.logging_level,
                    'formatter': 'standard',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'logs/credit_intel.log',
                    'maxBytes': self.deployment.log_file_max_bytes,
                    'backupCount': self.deployment.log_file_backup_count,
                }
            },
            'loggers': {
                '': {
                    'handlers': ['default', 'file'],
                    'level': self.deployment.logging_level,
                    'propagate': False
                }
            }
        }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config(environment: Optional[str] = None) -> Config:
    """Reload configuration with optional environment override."""
    global config
    config = Config(environment)
    return config
