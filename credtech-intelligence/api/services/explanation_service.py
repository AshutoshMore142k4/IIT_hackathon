# Service for model explainability
# File: /api/services/explanation_service.py

import numpy as np
import pandas as pd
import shap
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from functools import lru_cache
import pickle
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationDepth(Enum):
    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

class ExplanationType(Enum):
    GLOBAL = "global"
    LOCAL = "local"
    COUNTERFACTUAL = "counterfactual"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"

@dataclass
class ExplanationResult:
    explanation_type: ExplanationType
    confidence_score: float
    feature_contributions: Dict[str, float]
    plain_language_summary: str
    technical_details: Dict[str, Any]
    timestamp: datetime
    company_id: Optional[str] = None

class ExplanationService:
    """
    Comprehensive explainability service for credit scoring models
    Provides multiple types of explanations with caching and customization
    """
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self._shap_cache = {}
        self._explanation_templates = self._load_explanation_templates()
        self.feature_descriptions = self._load_feature_descriptions()
        
        # Initialize SHAP explainer (will be set when model is provided)
        self.explainer = None
        self.background_data = None
        
        logger.info("ExplanationService initialized")

    def initialize_explainer(self, model, background_data: pd.DataFrame):
        """Initialize SHAP explainer with model and background data"""
        try:
            # Use TreeExplainer for tree-based models, LinearExplainer for linear models
            if hasattr(model, 'predict_proba'):
                self.explainer = shap.Explainer(model, background_data)
            else:
                self.explainer = shap.LinearExplainer(model, background_data)
            
            self.background_data = background_data
            logger.info(f"SHAP explainer initialized with background data shape: {background_data.shape}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
            raise

    @lru_cache(maxsize=1000)
    def get_shap_explanations(self, model_hash: str, data_hash: str, customer_id: str) -> Dict[str, Any]:
        """
        Get SHAP explanations for a specific customer with caching
        
        Args:
            model_hash: Hash of the model for cache key
            data_hash: Hash of the data for cache key  
            customer_id: Customer identifier
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        try:
            cache_key = f"{model_hash}_{data_hash}_{customer_id}"
            
            if cache_key in self._shap_cache:
                logger.info(f"Returning cached SHAP explanation for customer {customer_id}")
                return self._shap_cache[cache_key]
            
            if self.explainer is None:
                raise ValueError("SHAP explainer not initialized. Call initialize_explainer() first.")
            
            # Get customer data (this would come from your data pipeline)
            customer_data = self._get_customer_data(customer_id)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(customer_data)
            
            # If binary classification, get positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class (default risk)
            
            # Create explanation dictionary
            feature_names = customer_data.columns if hasattr(customer_data, 'columns') else [f"feature_{i}" for i in range(len(shap_values))]
            
            explanation = {
                'customer_id': customer_id,
                'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                'feature_values': customer_data.values.tolist() if hasattr(customer_data, 'values') else customer_data,
                'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
                'base_value': self.explainer.expected_value,
                'prediction': float(np.sum(shap_values) + self.explainer.expected_value),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the explanation
            if len(self._shap_cache) < self.cache_size:
                self._shap_cache[cache_key] = explanation
            
            logger.info(f"Generated SHAP explanation for customer {customer_id}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations for customer {customer_id}: {str(e)}")
            raise

    def generate_counterfactuals(self, model, instance: pd.Series, target_score: float, 
                               max_changes: int = 3) -> Dict[str, Any]:
        """
        Generate counterfactual explanations showing what changes would achieve target score
        
        Args:
            model: Trained model
            instance: Single customer instance
            target_score: Desired credit score
            max_changes: Maximum number of features to change
            
        Returns:
            Dictionary with counterfactual scenarios
        """
        try:
            current_score = model.predict([instance.values])[0]
            
            if abs(current_score - target_score) < 0.01:
                return {
                    'message': 'Current score already matches target',
                    'current_score': float(current_score),
                    'target_score': float(target_score)
                }
            
            # Get SHAP values to identify most important features
            if self.explainer is None:
                raise ValueError("SHAP explainer not initialized")
            
            shap_values = self.explainer.shap_values([instance.values])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Sort features by absolute SHAP value
            feature_importance = [(i, abs(val)) for i, val in enumerate(shap_values[0])]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            counterfactuals = []
            
            # Generate counterfactuals by modifying top features
            for i in range(min(max_changes, len(feature_importance))):
                feature_idx = feature_importance[i][0]
                feature_name = instance.index[feature_idx]
                current_value = instance.iloc[feature_idx]
                
                # Generate realistic alternative values
                alternatives = self._generate_realistic_alternatives(
                    feature_name, current_value, target_score > current_score
                )
                
                for alt_value in alternatives:
                    modified_instance = instance.copy()
                    modified_instance.iloc[feature_idx] = alt_value
                    
                    predicted_score = model.predict([modified_instance.values])[0]
                    
                    counterfactual = {
                        'feature_changed': feature_name,
                        'original_value': float(current_value),
                        'new_value': float(alt_value),
                        'predicted_score': float(predicted_score),
                        'score_change': float(predicted_score - current_score),
                        'explanation': self._explain_counterfactual_change(
                            feature_name, current_value, alt_value, predicted_score - current_score
                        )
                    }
                    
                    counterfactuals.append(counterfactual)
                    
                    # Stop if we found a good counterfactual
                    if abs(predicted_score - target_score) < abs(current_score - target_score):
                        break
            
            # Sort by how close they get to target
            counterfactuals.sort(key=lambda x: abs(x['predicted_score'] - target_score))
            
            result = {
                'current_score': float(current_score),
                'target_score': float(target_score),
                'counterfactuals': counterfactuals[:5],  # Return top 5
                'confidence_score': self._calculate_counterfactual_confidence(counterfactuals),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Generated {len(counterfactuals)} counterfactual scenarios")
            return result
            
        except Exception as e:
            logger.error(f"Error generating counterfactuals: {str(e)}")
            raise

    def explain_score_change(self, old_data: pd.Series, new_data: pd.Series, 
                           time_period: str) -> Dict[str, Any]:
        """
        Explain why a credit score changed between two time periods
        
        Args:
            old_data: Customer data from previous period
            new_data: Customer data from current period
            time_period: Description of time period (e.g., "30 days")
            
        Returns:
            Dictionary explaining the score change
        """
        try:
            # Calculate SHAP values for both periods
            old_shap = self.explainer.shap_values([old_data.values])
            new_shap = self.explainer.shap_values([new_data.values])
            
            if isinstance(old_shap, list):
                old_shap = old_shap[1][0]  # Positive class, first instance
                new_shap = new_shap[1]
            else:
                old_shap = old_shap
                new_shap = new_shap
            
            # Calculate feature changes
            feature_changes = []
            for i, feature_name in enumerate(old_data.index):
                old_val = old_data.iloc[i]
                new_val = new_data.iloc[i]
                old_contrib = old_shap[i]
                new_contrib = new_shap[i]
                
                if abs(new_val - old_val) > 1e-6:  # Only include changed features
                    change = {
                        'feature_name': feature_name,
                        'old_value': float(old_val),
                        'new_value': float(new_val),
                        'value_change': float(new_val - old_val),
                        'old_contribution': float(old_contrib),
                        'new_contribution': float(new_contrib),
                        'contribution_change': float(new_contrib - old_contrib),
                        'explanation': self._explain_feature_change(
                            feature_name, old_val, new_val, new_contrib - old_contrib
                        )
                    }
                    feature_changes.append(change)
            
            # Sort by absolute contribution change
            feature_changes.sort(key=lambda x: abs(x['contribution_change']), reverse=True)
            
            # Calculate overall score change
            old_score = float(np.sum(old_shap) + self.explainer.expected_value)
            new_score = float(np.sum(new_shap) + self.explainer.expected_value)
            score_change = new_score - old_score
            
            result = {
                'time_period': time_period,
                'old_score': old_score,
                'new_score': new_score,
                'score_change': score_change,
                'direction': 'improved' if score_change > 0 else 'declined',
                'magnitude': 'significant' if abs(score_change) > 10 else 'moderate' if abs(score_change) > 5 else 'minor',
                'feature_changes': feature_changes[:10],  # Top 10 most impactful changes
                'summary': self._create_temporal_summary(score_change, feature_changes[:3]),
                'confidence_score': self._calculate_temporal_confidence(feature_changes),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Analyzed score change: {score_change:.2f} points over {time_period}")
            return result
            
        except Exception as e:
            logger.error(f"Error explaining score change: {str(e)}")
            raise

    def compare_to_peers(self, company_data: pd.Series, sector_benchmarks: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare company metrics to sector benchmarks
        
        Args:
            company_data: Current company financial data
            sector_benchmarks: Dictionary of sector average values
            
        Returns:
            Peer comparison analysis
        """
        try:
            comparisons = []
            
            for feature_name in company_data.index:
                if feature_name in sector_benchmarks:
                    company_val = company_data[feature_name]
                    sector_avg = sector_benchmarks[feature_name]
                    
                    # Calculate percentile and relative performance
                    difference = company_val - sector_avg
                    relative_perf = (difference / sector_avg) * 100 if sector_avg != 0 else 0
                    
                    comparison = {
                        'feature_name': feature_name,
                        'company_value': float(company_val),
                        'sector_average': float(sector_avg),
                        'difference': float(difference),
                        'relative_performance': float(relative_perf),
                        'performance_category': self._categorize_performance(relative_perf),
                        'explanation': self._explain_peer_comparison(
                            feature_name, company_val, sector_avg, relative_perf
                        )
                    }
                    comparisons.append(comparison)
            
            # Calculate overall peer ranking
            overall_score = self._calculate_peer_score(comparisons)
            
            result = {
                'overall_peer_ranking': overall_score,
                'ranking_category': self._categorize_peer_ranking(overall_score),
                'feature_comparisons': comparisons,
                'strengths': [c for c in comparisons if c['relative_performance'] > 10],
                'weaknesses': [c for c in comparisons if c['relative_performance'] < -10],
                'summary': self._create_peer_comparison_summary(overall_score, comparisons),
                'confidence_score': self._calculate_comparison_confidence(comparisons),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Generated peer comparison with overall score: {overall_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating peer comparison: {str(e)}")
            raise

    def create_plain_language_summary(self, technical_explanation: Dict[str, Any], 
                                    depth: ExplanationDepth = ExplanationDepth.SUMMARY) -> str:
        """
        Convert technical explanations to plain language
        
        Args:
            technical_explanation: Technical explanation dictionary
            depth: Level of detail for explanation
            
        Returns:
            Plain language explanation string
        """
        try:
            explanation_type = technical_explanation.get('type', 'general')
            
            if explanation_type == 'shap':
                return self._create_shap_summary(technical_explanation, depth)
            elif explanation_type == 'counterfactual':
                return self._create_counterfactual_summary(technical_explanation, depth)
            elif explanation_type == 'temporal':
                return self._create_temporal_summary(
                    technical_explanation.get('score_change', 0),
                    technical_explanation.get('feature_changes', [])
                )
            elif explanation_type == 'comparative':
                return self._create_peer_comparison_summary(
                    technical_explanation.get('overall_peer_ranking', 0),
                    technical_explanation.get('feature_comparisons', [])
                )
            else:
                return "Unable to generate plain language summary for this explanation type."
                
        except Exception as e:
            logger.error(f"Error creating plain language summary: {str(e)}")
            return "Error generating explanation summary."

    def get_feature_trend_analysis(self, feature_history: pd.DataFrame, 
                                 timeframe: str = "30d") -> Dict[str, Any]:
        """
        Analyze trends in feature values over time
        
        Args:
            feature_history: DataFrame with features over time
            timeframe: Analysis timeframe
            
        Returns:
            Feature trend analysis
        """
        try:
            trends = {}
            
            for feature in feature_history.columns:
                if feature == 'timestamp':
                    continue
                    
                values = feature_history[feature].dropna()
                
                if len(values) < 2:
                    continue
                
                # Calculate trend statistics
                trend_slope = self._calculate_trend_slope(values)
                volatility = values.std()
                recent_change = values.iloc[-1] - values.iloc[0] if len(values) > 1 else 0
                
                trend_analysis = {
                    'feature_name': feature,
                    'trend_direction': 'increasing' if trend_slope > 0.01 else 'decreasing' if trend_slope < -0.01 else 'stable',
                    'trend_strength': abs(trend_slope),
                    'volatility': float(volatility),
                    'recent_change': float(recent_change),
                    'current_value': float(values.iloc[-1]),
                    'average_value': float(values.mean()),
                    'explanation': self._explain_feature_trend(feature, trend_slope, volatility, recent_change)
                }
                
                trends[feature] = trend_analysis
            
            result = {
                'timeframe': timeframe,
                'trends': trends,
                'most_volatile_features': sorted(
                    trends.values(), 
                    key=lambda x: x['volatility'], 
                    reverse=True
                )[:5],
                'strongest_trends': sorted(
                    trends.values(), 
                    key=lambda x: x['trend_strength'], 
                    reverse=True
                )[:5],
                'summary': self._create_trend_summary(trends),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Analyzed trends for {len(trends)} features over {timeframe}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing feature trends: {str(e)}")
            raise

    # Helper methods
    def _get_customer_data(self, customer_id: str) -> pd.Series:
        """Get customer data by ID - implement based on your data pipeline"""
        # This would connect to your actual data source
        # For now, return dummy data
        return pd.Series([0.5, 0.3, 0.8, 0.2], index=['debt_ratio', 'roa', 'current_ratio', 'profit_margin'])

    def _generate_realistic_alternatives(self, feature_name: str, current_value: float, improve: bool) -> List[float]:
        """Generate realistic alternative values for counterfactuals"""
        alternatives = []
        
        # Define realistic ranges for different feature types
        if 'ratio' in feature_name.lower():
            step = 0.1 if improve else -0.1
            for i in range(1, 4):
                new_val = current_value + (step * i)
                if 0 <= new_val <= 2:  # Reasonable ratio range
                    alternatives.append(new_val)
        else:
            # For other features, use percentage changes
            multipliers = [1.1, 1.2, 1.3] if improve else [0.9, 0.8, 0.7]
            for mult in multipliers:
                alternatives.append(current_value * mult)
        
        return alternatives

    def _explain_counterfactual_change(self, feature_name: str, old_val: float, new_val: float, impact: float) -> str:
        """Create explanation for counterfactual change"""
        direction = "increase" if new_val > old_val else "decrease"
        impact_desc = "improve" if impact > 0 else "worsen"
        
        return f"If {feature_name} were to {direction} from {old_val:.3f} to {new_val:.3f}, " \
               f"the credit score would {impact_desc} by approximately {abs(impact):.2f} points."

    def _explain_feature_change(self, feature_name: str, old_val: float, new_val: float, contribution_change: float) -> str:
        """Explain how a feature change affected the score"""
        change_type = "increased" if new_val > old_val else "decreased"
        impact_type = "positively" if contribution_change > 0 else "negatively"
        
        return f"{feature_name} {change_type} from {old_val:.3f} to {new_val:.3f}, " \
               f"which {impact_type} impacted the credit score by {abs(contribution_change):.2f} points."

    def _explain_peer_comparison(self, feature_name: str, company_val: float, sector_avg: float, relative_perf: float) -> str:
        """Explain peer comparison for a feature"""
        comparison = "above" if company_val > sector_avg else "below"
        performance = "outperforming" if relative_perf > 0 else "underperforming"
        
        return f"{feature_name} is {abs(relative_perf):.1f}% {comparison} sector average, " \
               f"{performance} peers in this metric."

    def _categorize_performance(self, relative_perf: float) -> str:
        """Categorize relative performance"""
        if relative_perf > 20:
            return "excellent"
        elif relative_perf > 10:
            return "good"
        elif relative_perf > -10:
            return "average"
        elif relative_perf > -20:
            return "below_average"
        else:
            return "poor"

    def _categorize_peer_ranking(self, score: float) -> str:
        """Categorize overall peer ranking"""
        if score > 80:
            return "top_quartile"
        elif score > 60:
            return "above_average"
        elif score > 40:
            return "average"
        elif score > 20:
            return "below_average"
        else:
            return "bottom_quartile"

    def _calculate_peer_score(self, comparisons: List[Dict]) -> float:
        """Calculate overall peer performance score"""
        if not comparisons:
            return 50.0
        
        total_score = 0
        for comp in comparisons:
            # Convert relative performance to 0-100 score
            rel_perf = comp['relative_performance']
            if rel_perf > 0:
                score = min(100, 50 + rel_perf)
            else:
                score = max(0, 50 + rel_perf)
            total_score += score
        
        return total_score / len(comparisons)

    def _calculate_trend_slope(self, values: pd.Series) -> float:
        """Calculate trend slope for time series"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = values.values
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def _explain_feature_trend(self, feature: str, slope: float, volatility: float, recent_change: float) -> str:
        """Explain feature trend"""
        if abs(slope) < 0.01:
            trend_desc = "remained stable"
        elif slope > 0:
            trend_desc = "shown an increasing trend"
        else:
            trend_desc = "shown a decreasing trend"
        
        volatility_desc = "high" if volatility > 0.1 else "moderate" if volatility > 0.05 else "low"
        
        return f"{feature} has {trend_desc} with {volatility_desc} volatility " \
               f"and a recent change of {recent_change:.3f}."

    def _create_shap_summary(self, explanation: Dict, depth: ExplanationDepth) -> str:
        """Create SHAP explanation summary"""
        if depth == ExplanationDepth.SUMMARY:
            return "Credit score calculated based on key financial metrics and their relative importance."
        # Add more detailed summaries based on depth
        return "Detailed SHAP analysis available."

    def _create_counterfactual_summary(self, explanation: Dict, depth: ExplanationDepth) -> str:
        """Create counterfactual explanation summary"""
        return "Analysis shows potential changes that could improve credit score."

    def _create_temporal_summary(self, score_change: float, feature_changes: List[Dict]) -> str:
        """Create temporal change summary"""
        direction = "improved" if score_change > 0 else "declined"
        main_driver = feature_changes[0]['feature_name'] if feature_changes else "multiple factors"
        
        return f"Credit score {direction} by {abs(score_change):.2f} points, " \
               f"primarily driven by changes in {main_driver}."

    def _create_peer_comparison_summary(self, overall_score: float, comparisons: List[Dict]) -> str:
        """Create peer comparison summary"""
        ranking = self._categorize_peer_ranking(overall_score)
        return f"Overall peer performance ranks in {ranking.replace('_', ' ')} " \
               f"with a composite score of {overall_score:.1f}/100."

    def _create_trend_summary(self, trends: Dict) -> str:
        """Create trend analysis summary"""
        increasing_count = sum(1 for t in trends.values() if t['trend_direction'] == 'increasing')
        decreasing_count = sum(1 for t in trends.values() if t['trend_direction'] == 'decreasing')
        stable_count = len(trends) - increasing_count - decreasing_count
        
        return f"Analysis shows {increasing_count} improving trends, " \
               f"{decreasing_count} declining trends, and {stable_count} stable metrics."

    def _calculate_counterfactual_confidence(self, counterfactuals: List[Dict]) -> float:
        """Calculate confidence score for counterfactual explanations"""
        if not counterfactuals:
            return 0.0
        
        # Confidence based on consistency and realism of counterfactuals
        return min(0.9, len(counterfactuals) * 0.2)

    def _calculate_temporal_confidence(self, feature_changes: List[Dict]) -> float:
        """Calculate confidence score for temporal explanations"""
        if not feature_changes:
            return 0.0
        
        # Confidence based on number of significant changes
        significant_changes = sum(1 for fc in feature_changes if abs(fc['contribution_change']) > 1.0)
        return min(0.95, 0.5 + significant_changes * 0.1)

    def _calculate_comparison_confidence(self, comparisons: List[Dict]) -> float:
        """Calculate confidence score for peer comparisons"""
        if not comparisons:
            return 0.0
        
        # Confidence based on data completeness
        return min(0.9, len(comparisons) / 10.0)

    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates"""
        return {
            'default': "Analysis based on available financial data and market indicators.",
            'high_risk': "Multiple risk factors indicate elevated credit risk.",
            'low_risk': "Strong financial metrics support good creditworthiness."
        }

    def _load_feature_descriptions(self) -> Dict[str, str]:
        """Load feature descriptions for plain language explanations"""
        return {
            'debt_ratio': 'debt-to-equity ratio',
            'roa': 'return on assets',
            'current_ratio': 'current ratio (liquidity)',
            'profit_margin': 'profit margin',
            # Add more feature descriptions as needed
        }

    def export_explanation_to_pdf(self, explanation: Dict, filename: str) -> bool:
        """Export explanation to PDF report"""
        # Implement PDF generation logic
        logger.info(f"PDF export functionality to be implemented for {filename}")
        return True

    def get_explanation_confidence(self, explanation: Dict) -> float:
        """Calculate overall confidence score for explanation"""
        explanation_type = explanation.get('type', 'general')
        
        if explanation_type == 'shap':
            return 0.9  # SHAP explanations are generally reliable
        elif explanation_type == 'counterfactual':
            return self._calculate_counterfactual_confidence(explanation.get('counterfactuals', []))
        elif explanation_type == 'temporal':
            return self._calculate_temporal_confidence(explanation.get('feature_changes', []))
        elif explanation_type == 'comparative':
            return self._calculate_comparison_confidence(explanation.get('feature_comparisons', []))
        else:
            return 0.5  # Default moderate confidence

# Example usage and testing
if __name__ == "__main__":
    # Initialize service
    explanation_service = ExplanationService()
    
    # Example test (would require actual model and data in real implementation)
    print("ExplanationService initialized successfully")
    print(f"Cache size: {explanation_service.cache_size}")
    print(f"Available explanation types: {[e.value for e in ExplanationType]}")
    print(f"Available explanation depths: {[d.value for d in ExplanationDepth]}")
