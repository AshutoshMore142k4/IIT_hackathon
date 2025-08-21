# Feature engineering processor
# File: /data_pipeline/processors/feature_engineer.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering parameters"""
    # Time windows for rolling features
    short_window: int = 7
    medium_window: int = 30
    long_window: int = 90
    
    # Lag periods for predictive features
    lag_periods: List[int] = None
    
    # Missing data handling
    max_missing_ratio: float = 0.3
    imputation_method: str = "forward_fill"  # forward_fill, median, knn
    
    # Normalization methods
    financial_scaler: str = "robust"  # standard, robust, none
    market_scaler: str = "standard"
    news_scaler: str = "standard"
    
    # Sector groupings for relative metrics
    sector_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 7, 30]
        if self.sector_mapping is None:
            self.sector_mapping = {}

@dataclass
class FeatureImportance:
    """Track feature importance and quality metrics"""
    feature_name: str
    importance_score: float = 0.0
    missing_ratio: float = 0.0
    outlier_ratio: float = 0.0
    correlation_with_target: float = 0.0
    data_quality_score: float = 0.0

class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline for credit intelligence platform.
    
    Combines SEC EDGAR financial data, Yahoo Finance market data, and News sentiment
    to create a rich feature set for credit risk modeling.
    """
    
    def __init__(self, config: FeatureConfig = None, cache_dir: str = "./cache"):
        """
        Initialize feature engineer with configuration
        
        Args:
            config: Feature engineering configuration
            cache_dir: Directory for caching intermediate results
        """
        self.config = config or FeatureConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize scalers
        self.scalers = {
            'financial': RobustScaler() if self.config.financial_scaler == 'robust' else StandardScaler(),
            'market': StandardScaler(),
            'news': StandardScaler()
        }
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Cache for sector performance
        self.sector_cache = {}
        
        logger.info("FeatureEngineer initialized with configuration")
    
    def engineer_features(self, edgar_data: pd.DataFrame, yahoo_data: pd.DataFrame, 
                         news_data: pd.DataFrame, target_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            edgar_data: SEC EDGAR financial data
            yahoo_data: Yahoo Finance market data  
            news_data: News sentiment data
            target_data: Optional target variable for correlation analysis
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        try:
            # Step 1: Data validation and preprocessing
            edgar_data = self._validate_and_clean_data(edgar_data, 'edgar')
            yahoo_data = self._validate_and_clean_data(yahoo_data, 'yahoo')
            news_data = self._validate_and_clean_data(news_data, 'news')
            
            # Step 2: Merge data sources
            merged_data = self._merge_data_sources(edgar_data, yahoo_data, news_data)
            
            # Step 3: Generate base features
            features_df = self._generate_financial_features(merged_data)
            features_df = self._generate_market_features(features_df)
            features_df = self._generate_news_features(features_df)
            
            # Step 4: Generate derived features
            features_df = self._generate_derived_features(features_df)
            
            # Step 5: Create lag features
            features_df = self._create_lag_features(features_df)
            
            # Step 6: Handle missing data
            features_df = self._handle_missing_data(features_df)
            
            # Step 7: Normalize features
            features_df = self._normalize_features(features_df)
            
            # Step 8: Feature validation and importance calculation
            if target_data is not None:
                self._calculate_feature_importance(features_df, target_data)
            
            # Step 9: Cache results
            self._cache_features(features_df)
            
            logger.info(f"Feature engineering completed. Generated {len(features_df.columns)} features for {len(features_df)} observations")
            return features_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {e}")
            raise
    
    def _validate_and_clean_data(self, data: pd.DataFrame, source: str) -> pd.DataFrame:
        """Validate and clean input data"""
        if data.empty:
            logger.warning(f"Empty dataframe received for {source}")
            return data
        
        # Ensure required columns exist
        required_cols = {
            'edgar': ['ticker', 'date', 'revenue', 'total_debt', 'shareholders_equity', 'total_assets'],
            'yahoo': ['ticker', 'date', 'close', 'volume'],
            'news': ['ticker', 'date', 'sentiment_score']
        }
        
        missing_cols = set(required_cols[source]) - set(data.columns)
        if missing_cols:
            logger.warning(f"Missing required columns for {source}: {missing_cols}")
        
        # Convert date column
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        # Remove duplicates
        before_count = len(data)
        data = data.drop_duplicates(subset=['ticker', 'date'])
        after_count = len(data)
        
        if before_count != after_count:
            logger.info(f"Removed {before_count - after_count} duplicate records from {source}")
        
        return data.sort_values(['ticker', 'date'])
    
    def _merge_data_sources(self, edgar_data: pd.DataFrame, yahoo_data: pd.DataFrame, 
                           news_data: pd.DataFrame) -> pd.DataFrame:
        """Merge data from all sources with intelligent date alignment"""
        logger.info("Merging data sources")
        
        # Start with EDGAR data as base (quarterly frequency)
        merged = edgar_data.copy()
        
        # Merge Yahoo Finance data (daily frequency)
        # Use forward-fill to align quarterly financial data with daily market data
        if not yahoo_data.empty:
            merged = pd.merge_asof(
                merged.sort_values(['ticker', 'date']),
                yahoo_data.sort_values(['ticker', 'date']),
                on='date',
                by='ticker',
                direction='nearest',
                tolerance=pd.Timedelta(days=30)
            )
        
        # Merge news data (daily frequency)
        if not news_data.empty:
            merged = pd.merge_asof(
                merged.sort_values(['ticker', 'date']),
                news_data.sort_values(['ticker', 'date']),
                on='date',
                by='ticker',
                direction='nearest',
                tolerance=pd.Timedelta(days=7)
            )
        
        logger.info(f"Merged data shape: {merged.shape}")
        return merged
    
    def _generate_financial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate financial ratio and trend features"""
        logger.info("Generating financial features")
        
        features = data.copy()
        
        # Basic financial ratios
        features['debt_to_equity_ratio'] = self._safe_divide(features['total_debt'], features['shareholders_equity'])
        features['current_ratio'] = self._safe_divide(features['current_assets'], features['current_liabilities'])
        features['roa'] = self._safe_divide(features['net_income'], features['total_assets'])
        features['roe'] = self._safe_divide(features['net_income'], features['shareholders_equity'])
        features['asset_turnover'] = self._safe_divide(features['revenue'], features['total_assets'])
        features['debt_ratio'] = self._safe_divide(features['total_debt'], features['total_assets'])
        
        # Profitability metrics
        features['profit_margin'] = self._safe_divide(features['net_income'], features['revenue'])
        features['gross_margin'] = self._safe_divide(features['gross_profit'], features['revenue']) if 'gross_profit' in features.columns else np.nan
        
        # Liquidity metrics
        features['quick_ratio'] = self._safe_divide(
            features['current_assets'] - features.get('inventory', 0), 
            features['current_liabilities']
        )
        features['cash_ratio'] = self._safe_divide(features.get('cash', 0), features['current_liabilities'])
        
        # Growth metrics (Year-over-Year)
        for col in ['revenue', 'net_income', 'total_assets', 'shareholders_equity']:
            if col in features.columns:
                features[f'{col}_growth_yoy'] = features.groupby('ticker')[col].pct_change(periods=4)  # Quarterly to annual
        
        # Trend analysis (using rolling windows)
        for ticker_group in features.groupby('ticker'):
            ticker = ticker_group[0]
            ticker_data = ticker_group[1].sort_values('date')
            
            # Revenue trend (slope of last 4 quarters)
            if 'revenue' in ticker_data.columns:
                features.loc[features['ticker'] == ticker, 'revenue_trend'] = self._calculate_trend(ticker_data['revenue'], window=4)
            
            # Profit margin stability (rolling std dev)
            if 'profit_margin' in features.columns:
                features.loc[features['ticker'] == ticker, 'profit_margin_stability'] = (
                    ticker_data['profit_margin'].rolling(window=4, min_periods=2).std()
                )
        
        # Cash flow stability
        if 'cash_flow_operations' in features.columns:
            features['cash_flow_stability'] = features.groupby('ticker')['cash_flow_operations'].rolling(
                window=4, min_periods=2
            ).std().reset_index(level=0, drop=True)
        
        logger.info("Financial features generated")
        return features
    
    def _generate_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate market-based features"""
        logger.info("Generating market features")
        
        features = data.copy()
        
        if 'close' not in features.columns:
            logger.warning("No market price data available")
            return features
        
        # Price volatility metrics
        features['price_volatility_30d'] = features.groupby('ticker')['close'].rolling(
            window=30, min_periods=10
        ).std().reset_index(level=0, drop=True)
        
        features['price_volatility_90d'] = features.groupby('ticker')['close'].rolling(
            window=90, min_periods=30
        ).std().reset_index(level=0, drop=True)
        
        # Price momentum
        features['price_momentum_30d'] = features.groupby('ticker')['close'].pct_change(periods=30)
        features['price_momentum_90d'] = features.groupby('ticker')['close'].pct_change(periods=90)
        
        # Moving averages
        for window in [20, 50, 200]:
            features[f'ma_{window}d'] = features.groupby('ticker')['close'].rolling(
                window=window, min_periods=int(window/2)
            ).mean().reset_index(level=0, drop=True)
        
        # Relative strength indicators
        features['rsi_14d'] = features.groupby('ticker').apply(
            lambda x: self._calculate_rsi(x['close'], 14)
        ).reset_index(level=0, drop=True)
        
        # Volume features
        if 'volume' in features.columns:
            features['volume_trend'] = features.groupby('ticker')['volume'].rolling(
                window=30, min_periods=10
            ).mean().reset_index(level=0, drop=True)
            
            features['volume_volatility'] = features.groupby('ticker')['volume'].rolling(
                window=30, min_periods=10
            ).std().reset_index(level=0, drop=True)
        
        # Beta calculation (vs market benchmark - simplified)
        if len(features) > 100:  # Need sufficient data
            market_returns = features.groupby('date')['close'].mean().pct_change()
            
            for ticker in features['ticker'].unique():
                ticker_data = features[features['ticker'] == ticker].copy()
                if len(ticker_data) > 30:
                    ticker_returns = ticker_data['close'].pct_change()
                    aligned_market = market_returns.reindex(ticker_data['date']).fillna(0)
                    
                    # Calculate beta using covariance
                    covariance = np.cov(ticker_returns.fillna(0), aligned_market)[0, 1]
                    market_variance = np.var(aligned_market)
                    beta = covariance / market_variance if market_variance != 0 else 1.0
                    
                    features.loc[features['ticker'] == ticker, 'beta_coefficient'] = beta
        
        # Sector relative performance
        features = self._calculate_sector_relative_metrics(features)
        
        logger.info("Market features generated")
        return features
    
    def _generate_news_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate news sentiment and event-based features"""
        logger.info("Generating news features")
        
        features = data.copy()
        
        if 'sentiment_score' not in features.columns:
            logger.warning("No news sentiment data available")
            return features
        
        # Rolling sentiment averages
        for window in [7, 30, 90]:
            features[f'news_sentiment_{window}d'] = features.groupby('ticker')['sentiment_score'].rolling(
                window=window, min_periods=max(1, window//3)
            ).mean().reset_index(level=0, drop=True)
        
        # Sentiment volatility
        features['sentiment_volatility_30d'] = features.groupby('ticker')['sentiment_score'].rolling(
            window=30, min_periods=10
        ).std().reset_index(level=0, drop=True)
        
        # News volume metrics
        if 'news_volume' in features.columns:
            features['news_volume_trend'] = features.groupby('ticker')['news_volume'].rolling(
                window=30, min_periods=10
            ).mean().reset_index(level=0, drop=True)
            
            features['news_volume_spike'] = (
                features['news_volume'] > features['news_volume_trend'] * 2
            ).astype(float)
        
        # Event impact scoring
        event_columns = ['regulatory_risk_score', 'management_sentiment', 'earnings_sentiment']
        available_event_cols = [col for col in event_columns if col in features.columns]
        
        if available_event_cols:
            # Create weighted event impact score
            weights = np.array([0.4, 0.3, 0.3])[:len(available_event_cols)]
            weights = weights / weights.sum()  # Normalize
            
            features['event_impact_score'] = np.average(
                features[available_event_cols].fillna(0).values,
                weights=weights,
                axis=1
            )
        
        # Sentiment momentum
        features['sentiment_momentum'] = features.groupby('ticker')['sentiment_score'].diff(periods=7)
        
        # Sentiment vs market correlation
        if 'close' in features.columns:
            features['sentiment_price_correlation'] = features.groupby('ticker').apply(
                lambda x: x['sentiment_score'].rolling(30).corr(x['close'].pct_change()) if len(x) > 30 else np.nan
            ).reset_index(level=0, drop=True)
        
        logger.info("News features generated")
        return features
    
    def _generate_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate composite and derived features"""
        logger.info("Generating derived features")
        
        features = data.copy()
        
        # Credit Risk Composite Score
        credit_components = {
            'debt_to_equity_ratio': 0.25,
            'current_ratio': -0.20,  # Negative because higher is better
            'roa': -0.20,  # Negative because higher is better
            'roe': -0.15,  # Negative because higher is better
            'news_sentiment_30d': -0.10,  # Negative because positive sentiment is good
            'event_impact_score': 0.10
        }
        
        credit_risk_score = np.zeros(len(features))
        total_weight = 0
        
        for component, weight in credit_components.items():
            if component in features.columns:
                normalized_component = self._normalize_component(features[component])
                credit_risk_score += weight * normalized_component
                total_weight += abs(weight)
        
        if total_weight > 0:
            features['credit_risk_composite_score'] = credit_risk_score / total_weight
        
        # Financial Stress Indicator
        stress_conditions = []
        
        if 'debt_to_equity_ratio' in features.columns:
            stress_conditions.append((features['debt_to_equity_ratio'] > 1.5).astype(float))
        
        if 'current_ratio' in features.columns:
            stress_conditions.append((features['current_ratio'] < 1.0).astype(float))
        
        if 'roa' in features.columns:
            stress_conditions.append((features['roa'] < 0).astype(float))
        
        if 'cash_flow_operations' in features.columns:
            stress_conditions.append((features['cash_flow_operations'] < 0).astype(float))
        
        if stress_conditions:
            features['financial_stress_indicator'] = np.mean(stress_conditions, axis=0)
        
        # Market Confidence Index
        confidence_components = []
        
        if 'news_sentiment_7d' in features.columns:
            confidence_components.append(features['news_sentiment_7d'].fillna(0))
        
        if 'price_momentum_30d' in features.columns:
            confidence_components.append(features['price_momentum_30d'].fillna(0))
        
        if 'volume_trend' in features.columns:
            volume_normalized = self._normalize_component(features['volume_trend'])
            confidence_components.append(volume_normalized)
        
        if confidence_components:
            features['market_confidence_index'] = np.mean(confidence_components, axis=0)
        
        # Liquidity Risk Score
        if all(col in features.columns for col in ['current_ratio', 'quick_ratio', 'cash_ratio']):
            features['liquidity_risk_score'] = (
                0.4 * (2.0 - features['current_ratio'].clip(0, 4)) / 4.0 +
                0.4 * (1.5 - features['quick_ratio'].clip(0, 3)) / 3.0 +
                0.2 * (0.5 - features['cash_ratio'].clip(0, 1)) / 1.0
            )
        
        # Profitability Trend Score
        profitability_cols = ['revenue_growth_yoy', 'profit_margin', 'roa', 'roe']
        available_prof_cols = [col for col in profitability_cols if col in features.columns]
        
        if available_prof_cols:
            prof_data = features[available_prof_cols].fillna(0)
            features['profitability_trend_score'] = prof_data.mean(axis=1)
        
        logger.info("Derived features generated")
        return features
    
    def _create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for predictive modeling"""
        logger.info("Creating lag features")
        
        features = data.copy()
        
        # Define feature groups for lagging
        financial_features = [col for col in features.columns if any(
            keyword in col.lower() for keyword in ['ratio', 'margin', 'growth', 'roa', 'roe']
        )]
        
        market_features = [col for col in features.columns if any(
            keyword in col.lower() for keyword in ['price', 'volume', 'momentum', 'volatility', 'beta']
        )]
        
        news_features = [col for col in features.columns if any(
            keyword in col.lower() for keyword in ['sentiment', 'news', 'event']
        )]
        
        # Create lags for each feature group
        feature_groups = {
            'financial': financial_features,
            'market': market_features,
            'news': news_features
        }
        
        for group_name, feature_list in feature_groups.items():
            for lag_period in self.config.lag_periods:
                for feature in feature_list:
                    if feature in features.columns:
                        lag_col_name = f"{feature}_lag_{lag_period}d"
                        features[lag_col_name] = features.groupby('ticker')[feature].shift(lag_period)
        
        logger.info(f"Created lag features for {len(self.config.lag_periods)} time periods")
        return features
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data with multiple strategies"""
        logger.info("Handling missing data")
        
        features = data.copy()
        
        # Calculate missing ratios
        missing_ratios = features.isnull().sum() / len(features)
        high_missing_cols = missing_ratios[missing_ratios > self.config.max_missing_ratio].index
        
        if len(high_missing_cols) > 0:
            logger.warning(f"Dropping {len(high_missing_cols)} columns with >30% missing data")
            features = features.drop(columns=high_missing_cols)
        
        # Imputation strategies by feature type
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        if self.config.imputation_method == "forward_fill":
            # Forward fill within each ticker group
            for col in numeric_cols:
                features[col] = features.groupby('ticker')[col].fillna(method='ffill')
            
            # Backward fill remaining
            for col in numeric_cols:
                features[col] = features.groupby('ticker')[col].fillna(method='bfill')
        
        elif self.config.imputation_method == "median":
            # Median imputation by ticker
            for ticker in features['ticker'].unique():
                ticker_mask = features['ticker'] == ticker
                ticker_data = features.loc[ticker_mask, numeric_cols]
                features.loc[ticker_mask, numeric_cols] = ticker_data.fillna(ticker_data.median())
        
        elif self.config.imputation_method == "knn":
            # KNN imputation (computationally expensive, use sparingly)
            knn_imputer = KNNImputer(n_neighbors=5)
            for ticker in features['ticker'].unique():
                ticker_mask = features['ticker'] == ticker
                ticker_data = features.loc[ticker_mask, numeric_cols]
                if len(ticker_data) > 10:  # Need sufficient data for KNN
                    imputed_data = knn_imputer.fit_transform(ticker_data)
                    features.loc[ticker_mask, numeric_cols] = imputed_data
        
        # Fill any remaining NaN with 0
        features[numeric_cols] = features[numeric_cols].fillna(0)
        
        logger.info("Missing data handling completed")
        return features
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using appropriate scalers"""
        logger.info("Normalizing features")
        
        features = data.copy()
        
        # Define feature groups for different normalization strategies
        financial_features = [col for col in features.columns if any(
            keyword in col.lower() for keyword in ['ratio', 'margin', 'growth', 'roa', 'roe', 'debt', 'asset']
        )]
        
        market_features = [col for col in features.columns if any(
            keyword in col.lower() for keyword in ['price', 'volume', 'momentum', 'volatility', 'beta', 'rsi']
        )]
        
        news_features = [col for col in features.columns if any(
            keyword in col.lower() for keyword in ['sentiment', 'news', 'event']
        )]
        
        # Apply scalers to each group
        feature_groups = {
            'financial': (financial_features, self.scalers['financial']),
            'market': (market_features, self.scalers['market']),
            'news': (news_features, self.scalers['news'])
        }
        
        for group_name, (feature_list, scaler) in feature_groups.items():
            available_features = [f for f in feature_list if f in features.columns]
            
            if available_features and self.config.__dict__.get(f'{group_name}_scaler') != 'none':
                # Fit and transform
                scaled_data = scaler.fit_transform(features[available_features])
                features[available_features] = scaled_data
                
                logger.info(f"Normalized {len(available_features)} {group_name} features")
        
        return features
    
    def _calculate_feature_importance(self, features: pd.DataFrame, target: pd.DataFrame):
        """Calculate feature importance and quality metrics"""
        logger.info("Calculating feature importance")
        
        # Merge features with target
        merged = features.merge(target, on=['ticker', 'date'], how='inner')
        
        if 'target' not in merged.columns:
            logger.warning("No target column found for importance calculation")
            return
        
        feature_cols = [col for col in merged.columns if col not in ['ticker', 'date', 'target']]
        
        for feature in feature_cols:
            # Calculate correlation with target
            correlation = merged[feature].corr(merged['target'])
            
            # Calculate missing ratio
            missing_ratio = merged[feature].isnull().sum() / len(merged)
            
            # Calculate outlier ratio (values beyond 3 standard deviations)
            z_scores = np.abs((merged[feature] - merged[feature].mean()) / merged[feature].std())
            outlier_ratio = (z_scores > 3).sum() / len(merged)
            
            # Calculate data quality score
            quality_score = (1 - missing_ratio) * (1 - outlier_ratio) * abs(correlation)
            
            # Store feature importance
            self.feature_importance[feature] = FeatureImportance(
                feature_name=feature,
                importance_score=abs(correlation),
                missing_ratio=missing_ratio,
                outlier_ratio=outlier_ratio,
                correlation_with_target=correlation,
                data_quality_score=quality_score
            )
        
        logger.info(f"Calculated importance for {len(feature_cols)} features")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance summary"""
        if not self.feature_importance:
            logger.warning("No feature importance calculated yet")
            return pd.DataFrame()
        
        importance_data = []
        for feature_name, importance in self.feature_importance.items():
            importance_data.append({
                'feature_name': importance.feature_name,
                'importance_score': importance.importance_score,
                'missing_ratio': importance.missing_ratio,
                'outlier_ratio': importance.outlier_ratio,
                'correlation_with_target': importance.correlation_with_target,
                'data_quality_score': importance.data_quality_score
            })
        
        return pd.DataFrame(importance_data).sort_values('importance_score', ascending=False)
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary statistics of generated features"""
        return {
            'total_features': len(self.feature_importance),
            'avg_importance': np.mean([imp.importance_score for imp in self.feature_importance.values()]),
            'high_quality_features': len([imp for imp in self.feature_importance.values() if imp.data_quality_score > 0.7]),
            'features_with_high_missing': len([imp for imp in self.feature_importance.values() if imp.missing_ratio > 0.2])
        }
    
    # Helper methods
    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Safe division avoiding division by zero"""
        return np.where(denominator != 0, numerator / denominator, np.nan)
    
    def _normalize_component(self, series: pd.Series) -> pd.Series:
        """Normalize component to 0-1 range"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return np.zeros_like(series)
        return (series - min_val) / (max_val - min_val)
    
    def _calculate_trend(self, series: pd.Series, window: int) -> float:
        """Calculate trend as slope of linear regression"""
        if len(series) < window:
            return np.nan
        
        y = series.tail(window).values
        x = np.arange(len(y))
        
        if len(x) < 2:
            return np.nan
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_sector_relative_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate sector-relative performance metrics"""
        if not self.config.sector_mapping:
            return data
        
        features = data.copy()
        
        # Add sector information
        features['sector'] = features['ticker'].map(self.config.sector_mapping)
        
        # Calculate sector averages for key metrics
        sector_metrics = ['close', 'roa', 'roe', 'debt_to_equity_ratio']
        available_metrics = [m for m in sector_metrics if m in features.columns]
        
        for metric in available_metrics:
            sector_avg = features.groupby(['date', 'sector'])[metric].transform('mean')
            features[f'{metric}_vs_sector'] = features[metric] / sector_avg
        
        return features.drop(columns=['sector'])  # Remove temporary sector column
    
    def _cache_features(self, features: pd.DataFrame):
        """Cache generated features for faster recomputation"""
        try:
            cache_file = self.cache_dir / f"features_cache_{datetime.now().strftime('%Y%m%d')}.parquet"
            features.to_parquet(cache_file, compression='snappy')
            logger.info(f"Features cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not cache features: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize feature engineer
    config = FeatureConfig(
        short_window=7,
        medium_window=30,
        long_window=90,
        lag_periods=[1, 7, 30],
        imputation_method="forward_fill",
        financial_scaler="robust",
        sector_mapping={
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'JPM': 'Financial',
            'XOM': 'Energy'
        }
    )
    
    engineer = FeatureEngineer(config)
    
    # Create sample data for testing
    print("Creating sample data for feature engineering demo...")
    
    # Sample EDGAR data
    dates = pd.date_range('2023-01-01', periods=12, freq='QS')
    tickers = ['AAPL', 'MSFT', 'JPM']
    
    edgar_data = []
    for ticker in tickers:
        for date in dates:
            edgar_data.append({
                'ticker': ticker,
                'date': date,
                'revenue': np.random.uniform(50000, 100000),
                'total_debt': np.random.uniform(20000, 40000),
                'shareholders_equity': np.random.uniform(30000, 60000),
                'total_assets': np.random.uniform(80000, 150000),
                'current_assets': np.random.uniform(40000, 80000),
                'current_liabilities': np.random.uniform(20000, 40000),
                'net_income': np.random.uniform(5000, 15000),
                'cash_flow_operations': np.random.uniform(8000, 20000)
            })
    
    edgar_df = pd.DataFrame(edgar_data)
    
    # Sample Yahoo Finance data
    daily_dates = pd.date_range('2023-01-01', periods=365, freq='D')
    yahoo_data = []
    
    for ticker in tickers:
        base_price = np.random.uniform(100, 200)
        for i, date in enumerate(daily_dates):
            # Add some random walk to price
            price_change = np.random.normal(0, 0.02)
            base_price *= (1 + price_change)
            
            yahoo_data.append({
                'ticker': ticker,
                'date': date,
                'close': base_price,
                'volume': np.random.uniform(1000000, 10000000)
            })
    
    yahoo_df = pd.DataFrame(yahoo_data)
    
    # Sample news data
    news_data = []
    for ticker in tickers:
        for date in daily_dates[::7]:  # Weekly news
            news_data.append({
                'ticker': ticker,
                'date': date,
                'sentiment_score': np.random.uniform(-1, 1),
                'news_volume': np.random.randint(1, 20),
                'regulatory_risk_score': np.random.uniform(0, 1),
                'management_sentiment': np.random.uniform(-1, 1)
            })
    
    news_df = pd.DataFrame(news_data)
    
    # Generate features
    print("Running feature engineering pipeline...")
    features = engineer.engineer_features(edgar_df, yahoo_df, news_df)
    
    print(f"\n✅ Feature Engineering Completed!")
    print(f"Generated {len(features.columns)} features for {len(features)} observations")
    print(f"Features include:")
    
    feature_types = {
        'Financial': [col for col in features.columns if any(kw in col.lower() for kw in ['ratio', 'roa', 'roe', 'margin'])],
        'Market': [col for col in features.columns if any(kw in col.lower() for kw in ['price', 'volume', 'momentum', 'volatility'])],
        'News': [col for col in features.columns if any(kw in col.lower() for kw in ['sentiment', 'news', 'event'])],
        'Derived': [col for col in features.columns if any(kw in col.lower() for kw in ['composite', 'stress', 'confidence'])]
    }
    
    for feature_type, feature_list in feature_types.items():
        if feature_list:
            print(f"  {feature_type}: {len(feature_list)} features")
            print(f"    Examples: {feature_list[:3]}")
    
    # Show sample of generated features
    print(f"\nSample of generated features:")
    print(features[['ticker', 'date', 'debt_to_equity_ratio', 'credit_risk_composite_score', 
                   'news_sentiment_7d', 'market_confidence_index']].head(10))
    
    print("\n🎯 Feature engineering pipeline ready for credit intelligence platform!")
