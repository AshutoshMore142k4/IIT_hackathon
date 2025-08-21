# File: /api/routes/realtime.py

from flask import Blueprint, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
import redis
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid

# Import custom services
from ..services.model_service import ModelService
from ..services.explanation_service import ExplanationService
from ..services.data_collector import DataCollector
from ..services.news_analyzer import NewsAnalyzer
from ..utils.cache_manager import CacheManager
from ..utils.rate_limiter import RateLimiter
from ..utils.error_handler import APIError, handle_api_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize blueprint
realtime_bp = Blueprint('realtime', __name__, url_prefix='/api/v1/realtime')

# Initialize services
model_service = ModelService()
explanation_service = ExplanationService()
data_collector = DataCollector()
news_analyzer = NewsAnalyzer()
cache_manager = CacheManager()
rate_limiter = RateLimiter()

# Initialize rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour", "10 per minute"]
)

# Initialize Redis for caching and session management
try:
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True,
        socket_timeout=5
    )
    redis_client.ping()
    logger.info("Redis connection established for realtime API")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

# WebSocket configuration
socketio = SocketIO(
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True
)

# Performance monitoring decorator
def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
            
            # Store performance metrics
            if redis_client:
                metric_key = f"performance:{func.__name__}:{datetime.now().strftime('%H')}"
                redis_client.lpush(metric_key, execution_time)
                redis_client.expire(metric_key, 3600)  # 1 hour TTL
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    return wrapper

# Cache decorator for expensive operations
def cache_response(ttl=300, key_prefix="api"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not redis_client:
                return func(*args, **kwargs)
            
            # Generate cache key from function name and arguments
            cache_key = f"{key_prefix}:{func.__name__}:{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
            logger.debug(f"Cached result for {cache_key}")
            
            return result
        return wrapper
    return decorator

# Validation helpers
def validate_ticker(ticker: str) -> bool:
    """Validate ticker symbol format"""
    if not ticker or not isinstance(ticker, str):
        return False
    return len(ticker.strip().upper()) <= 10 and ticker.isalnum()

def validate_alert_data(data: Dict) -> Tuple[bool, str]:
    """Validate alert creation data"""
    required_fields = ['ticker', 'alert_type', 'threshold', 'condition']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    if not validate_ticker(data['ticker']):
        return False, "Invalid ticker symbol"
    
    if data['alert_type'] not in ['score_change', 'threshold_breach', 'news_alert', 'volatility']:
        return False, "Invalid alert type"
    
    return True, "Valid"

# =====================================================
# REAL-TIME CREDIT SCORES ENDPOINT
# =====================================================

@realtime_bp.route('/scores/<ticker>', methods=['GET'])
@limiter.limit("30 per minute")
@monitor_performance
@cache_response(ttl=180, key_prefix="scores")  # 3-minute cache
def get_realtime_score(ticker: str):
    """
    Get current credit score with real-time data and trend analysis
    
    Returns:
        JSON with score, confidence, timestamp, trend_direction, and metadata
    """
    try:
        # Validate ticker
        if not validate_ticker(ticker):
            raise APIError("Invalid ticker symbol", status_code=400)
        
        ticker = ticker.upper().strip()
        
        logger.info(f"Fetching real-time credit score for {ticker}")
        
        # Get current model prediction with uncertainty
        current_data = data_collector.get_latest_features(ticker)
        if not current_data:
            raise APIError(f"No data available for ticker {ticker}", status_code=404)
        
        # Generate score with uncertainty quantification
        prediction_result = model_service.predict_with_uncertainty([current_data])
        current_score = prediction_result['predictions'][0]
        confidence = prediction_result['confidence'][0]
        
        # Get historical scores for trend analysis
        historical_scores = cache_manager.get_score_history(ticker, days=30)
        
        # Calculate trend direction
        trend_direction = "neutral"
        trend_magnitude = 0.0
        
        if len(historical_scores) >= 2:
            recent_scores = [s['score'] for s in historical_scores[-7:]]  # Last 7 days
            older_scores = [s['score'] for s in historical_scores[-14:-7]]  # Previous 7 days
            
            if recent_scores and older_scores:
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                trend_magnitude = recent_avg - older_avg
                
                if trend_magnitude > 2:
                    trend_direction = "improving"
                elif trend_magnitude < -2:
                    trend_direction = "declining"
        
        # Get additional context
        market_data = data_collector.get_market_context(ticker)
        news_sentiment = news_analyzer.get_recent_sentiment(ticker, days_back=1)
        
        # Prepare response
        response = {
            "ticker": ticker,
            "score": round(float(current_score), 2),
            "confidence": round(float(confidence), 3),
            "last_update": datetime.now().isoformat(),
            "trend_direction": trend_direction,
            "trend_magnitude": round(trend_magnitude, 2),
            "risk_level": _get_risk_level(current_score),
            "metadata": {
                "data_freshness": current_data.get('data_age_minutes', 0),
                "model_version": model_service.get_model_version(),
                "market_cap": market_data.get('market_cap'),
                "sector": market_data.get('sector', 'Unknown'),
                "recent_news_sentiment": news_sentiment.get('average_sentiment', 0.0),
                "volatility_30d": market_data.get('volatility_30d'),
                "last_earnings_date": market_data.get('last_earnings_date')
            },
            "alerts": _check_score_alerts(ticker, current_score, trend_magnitude)
        }
        
        # Store score in history for trend analysis
        cache_manager.store_score_history(ticker, current_score, confidence)
        
        # Emit real-time update via WebSocket
        socketio.emit('score_update', {
            'ticker': ticker,
            'score': current_score,
            'timestamp': datetime.now().isoformat()
        }, room=f"scores_{ticker}")
        
        logger.info(f"Successfully returned real-time score for {ticker}: {current_score}")
        return jsonify(response)
        
    except APIError as e:
        logger.error(f"API error for {ticker}: {e.message}")
        return handle_api_error(e)
    except Exception as e:
        logger.error(f"Unexpected error getting score for {ticker}: {str(e)}")
        return handle_api_error(APIError("Internal server error", status_code=500))

def _get_risk_level(score: float) -> str:
    """Convert numerical score to risk level"""
    if score >= 80:
        return "low"
    elif score >= 60:
        return "medium"
    elif score >= 40:
        return "high"
    else:
        return "very_high"

def _check_score_alerts(ticker: str, current_score: float, trend_magnitude: float) -> List[Dict]:
    """Check if score triggers any alerts"""
    alerts = []
    
    # Check for significant score changes
    if abs(trend_magnitude) > 5:
        alerts.append({
            "type": "significant_change",
            "severity": "high" if abs(trend_magnitude) > 10 else "medium",
            "message": f"Credit score {'improved' if trend_magnitude > 0 else 'declined'} by {abs(trend_magnitude):.1f} points"
        })
    
    # Check for threshold breaches
    if current_score < 50:
        alerts.append({
            "type": "threshold_breach",
            "severity": "critical",
            "message": "Credit score below investment grade threshold"
        })
    
    return alerts

# =====================================================
# ALERTS MANAGEMENT ENDPOINTS
# =====================================================

@realtime_bp.route('/alerts', methods=['GET'])
@limiter.limit("60 per hour")
@monitor_performance
def get_active_alerts():
    """Get all active alerts for the user's watchlist"""
    try:
        user_id = request.args.get('user_id', 'default_user')
        
        # Get user's watchlist
        watchlist = _get_user_watchlist(user_id)
        
        active_alerts = []
        
        for ticker in watchlist:
            # Get ticker-specific alerts
            ticker_alerts = _get_ticker_alerts(ticker)
            
            # Check current conditions
            current_conditions = _check_alert_conditions(ticker)
            
            for alert in ticker_alerts:
                if _evaluate_alert_condition(alert, current_conditions):
                    alert_data = {
                        "id": alert['id'],
                        "ticker": ticker,
                        "type": alert['alert_type'],
                        "message": alert['message'],
                        "severity": alert['severity'],
                        "triggered_at": datetime.now().isoformat(),
                        "threshold": alert['threshold'],
                        "current_value": current_conditions.get(alert['condition_field'])
                    }
                    active_alerts.append(alert_data)
        
        # Sort by severity and time
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        active_alerts.sort(key=lambda x: (severity_order.get(x['severity'], 4), x['triggered_at']))
        
        response = {
            "alerts": active_alerts,
            "total_count": len(active_alerts),
            "critical_count": len([a for a in active_alerts if a['severity'] == 'critical']),
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(f"Retrieved {len(active_alerts)} active alerts for user {user_id}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {str(e)}")
        return handle_api_error(APIError("Error retrieving alerts", status_code=500))

@realtime_bp.route('/alerts', methods=['POST'])
@limiter.limit("20 per hour")
@monitor_performance
def create_alert_rule():
    """Create a new alert rule"""
    try:
        data = request.get_json()
        
        if not data:
            raise APIError("No data provided", status_code=400)
        
        # Validate alert data
        is_valid, error_message = validate_alert_data(data)
        if not is_valid:
            raise APIError(error_message, status_code=400)
        
        user_id = data.get('user_id', 'default_user')
        
        # Create alert rule
        alert_id = str(uuid.uuid4())
        alert_rule = {
            "id": alert_id,
            "user_id": user_id,
            "ticker": data['ticker'].upper(),
            "alert_type": data['alert_type'],
            "threshold": float(data['threshold']),
            "condition": data['condition'],  # 'greater_than', 'less_than', 'equals'
            "condition_field": data.get('condition_field', 'score'),
            "severity": data.get('severity', 'medium'),
            "enabled": True,
            "created_at": datetime.now().isoformat(),
            "message_template": data.get('message', "Alert triggered for {ticker}")
        }
        
        # Store alert rule
        _store_alert_rule(alert_rule)
        
        logger.info(f"Created alert rule {alert_id} for {data['ticker']}")
        
        return jsonify({
            "alert_id": alert_id,
            "message": "Alert rule created successfully",
            "alert_rule": alert_rule
        }), 201
        
    except APIError as e:
        return handle_api_error(e)
    except Exception as e:
        logger.error(f"Error creating alert rule: {str(e)}")
        return handle_api_error(APIError("Error creating alert", status_code=500))

@realtime_bp.route('/alerts/<alert_id>', methods=['DELETE'])
@limiter.limit("30 per hour")
@monitor_performance
def delete_alert_rule(alert_id: str):
    """Delete an alert rule"""
    try:
        if not alert_id:
            raise APIError("Alert ID required", status_code=400)
        
        # Remove alert rule
        success = _delete_alert_rule(alert_id)
        
        if not success:
            raise APIError("Alert rule not found", status_code=404)
        
        logger.info(f"Deleted alert rule {alert_id}")
        
        return jsonify({
            "message": "Alert rule deleted successfully",
            "alert_id": alert_id
        })
        
    except APIError as e:
        return handle_api_error(e)
    except Exception as e:
        logger.error(f"Error deleting alert rule: {str(e)}")
        return handle_api_error(APIError("Error deleting alert", status_code=500))

# =====================================================
# NEWS & SENTIMENT ENDPOINT
# =====================================================

@realtime_bp.route('/news/<ticker>', methods=['GET'])
@limiter.limit("40 per hour")
@monitor_performance
@cache_response(ttl=600, key_prefix="news")  # 10-minute cache
def get_realtime_news(ticker: str):
    """Get recent news with sentiment analysis for a ticker"""
    try:
        if not validate_ticker(ticker):
            raise APIError("Invalid ticker symbol", status_code=400)
        
        ticker = ticker.upper().strip()
        days_back = int(request.args.get('days', 7))
        limit = int(request.args.get('limit', 20))
        
        logger.info(f"Fetching news and sentiment for {ticker}")
        
        # Get news articles
        news_articles = news_analyzer.get_company_news(ticker, days_back=days_back)[:limit]
        
        if not news_articles:
            return jsonify({
                "ticker": ticker,
                "articles": [],
                "sentiment_summary": {
                    "average_sentiment": 0.0,
                    "sentiment_trend": "neutral",
                    "total_articles": 0
                },
                "impact_assessment": "No recent news available",
                "last_updated": datetime.now().isoformat()
            })
        
        # Analyze sentiment for each article
        analyzed_articles = []
        sentiment_scores = []
        
        for article in news_articles:
            # Get sentiment analysis
            sentiment_data = news_analyzer.analyze_article_sentiment(article)
            
            # Calculate impact on credit risk
            risk_impact = news_analyzer.assess_credit_risk_impact(article)
            
            analyzed_article = {
                "title": article.get('title', ''),
                "published_at": article.get('publishedAt', ''),
                "source": article.get('source', {}).get('name', 'Unknown'),
                "url": article.get('url', ''),
                "sentiment": {
                    "score": sentiment_data.get('score', 0.0),
                    "label": sentiment_data.get('label', 'neutral'),
                    "confidence": sentiment_data.get('confidence', 0.0)
                },
                "risk_impact": {
                    "level": risk_impact.get('level', 'neutral'),
                    "score": risk_impact.get('score', 0.0),
                    "reasoning": risk_impact.get('reasoning', '')
                },
                "key_events": article.get('financial_events', []),
                "relevance_score": article.get('relevance_score', 0.0)
            }
            
            analyzed_articles.append(analyzed_article)
            sentiment_scores.append(sentiment_data.get('score', 0.0))
        
        # Calculate sentiment summary
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        # Determine sentiment trend
        if len(sentiment_scores) >= 2:
            recent_sentiment = sum(sentiment_scores[-3:]) / len(sentiment_scores[-3:])
            older_sentiment = sum(sentiment_scores[:-3]) / len(sentiment_scores[:-3]) if len(sentiment_scores) > 3 else recent_sentiment
            
            if recent_sentiment > older_sentiment + 0.1:
                sentiment_trend = "improving"
            elif recent_sentiment < older_sentiment - 0.1:
                sentiment_trend = "declining"
            else:
                sentiment_trend = "stable"
        else:
            sentiment_trend = "neutral"
        
        # Generate impact assessment
        impact_assessment = _generate_impact_assessment(analyzed_articles, average_sentiment)
        
        response = {
            "ticker": ticker,
            "articles": analyzed_articles,
            "sentiment_summary": {
                "average_sentiment": round(average_sentiment, 3),
                "sentiment_trend": sentiment_trend,
                "total_articles": len(analyzed_articles),
                "positive_count": len([s for s in sentiment_scores if s > 0.1]),
                "negative_count": len([s for s in sentiment_scores if s < -0.1]),
                "neutral_count": len([s for s in sentiment_scores if -0.1 <= s <= 0.1])
            },
            "impact_assessment": impact_assessment,
            "last_updated": datetime.now().isoformat(),
            "data_freshness_minutes": 0  # Real-time data
        }
        
        logger.info(f"Retrieved {len(analyzed_articles)} news articles for {ticker}")
        return jsonify(response)
        
    except APIError as e:
        return handle_api_error(e)
    except Exception as e:
        logger.error(f"Error getting news for {ticker}: {str(e)}")
        return handle_api_error(APIError("Error retrieving news data", status_code=500))

def _generate_impact_assessment(articles: List[Dict], avg_sentiment: float) -> str:
    """Generate plain-language impact assessment from news analysis"""
    
    high_impact_events = [a for a in articles if a['risk_impact']['level'] in ['high', 'critical']]
    
    if not articles:
        return "No recent news impact detected."
    
    if high_impact_events:
        return f"Significant credit-relevant events detected. {len(high_impact_events)} high-impact news items may affect creditworthiness."
    elif avg_sentiment > 0.2:
        return "Positive news sentiment detected. Overall market confidence appears supportive of credit profile."
    elif avg_sentiment < -0.2:
        return "Negative news sentiment detected. Market concerns may create headwinds for credit quality."
    else:
        return "Mixed news sentiment with no clear directional impact on credit risk."

# =====================================================
# EXPLANATIONS ENDPOINT
# =====================================================

@realtime_bp.route('/explanations/<ticker>', methods=['GET'])
@limiter.limit("25 per hour")
@monitor_performance
@cache_response(ttl=240, key_prefix="explanations")  # 4-minute cache
def get_realtime_explanations(ticker: str):
    """Get real-time explanations for current credit score"""
    try:
        if not validate_ticker(ticker):
            raise APIError("Invalid ticker symbol", status_code=400)
        
        ticker = ticker.upper().strip()
        explanation_type = request.args.get('type', 'comprehensive')  # 'summary', 'detailed', 'comprehensive'
        
        logger.info(f"Generating explanations for {ticker}")
        
        # Get current data and prediction
        current_data = data_collector.get_latest_features(ticker)
        if not current_data:
            raise APIError(f"No data available for ticker {ticker}", status_code=404)
        
        # Get model prediction
        prediction_result = model_service.predict_with_uncertainty([current_data])
        current_score = prediction_result['predictions'][0]
        
        # Generate SHAP explanations
        shap_explanations = explanation_service.get_shap_explanations(
            model=model_service.get_model(),
            X=[current_data],
            feature_names=model_service.get_feature_names()
        )
        
        # Generate counterfactual explanations
        counterfactuals = explanation_service.generate_counterfactuals(
            model=model_service.get_model(),
            instance=current_data,
            target_scores=[current_score + 10, current_score - 10]
        )
        
        # Get temporal explanations (score change over time)
        temporal_explanation = explanation_service.explain_score_change(
            ticker=ticker,
            timeframe_days=30
        )
        
        # Generate plain-language summary
        plain_text_summary = explanation_service.create_plain_language_summary(
            shap_values=shap_explanations,
            score=current_score,
            ticker=ticker
        )
        
        # Get peer comparison context
        peer_comparison = explanation_service.compare_to_peers(
            ticker=ticker,
            current_data=current_data,
            current_score=current_score
        )
        
        response = {
            "ticker": ticker,
            "current_score": round(float(current_score), 2),
            "explanation_type": explanation_type,
            "feature_contributions": {
                "top_positive": shap_explanations.get('top_positive', []),
                "top_negative": shap_explanations.get('top_negative', []),
                "feature_values": shap_explanations.get('feature_values', {}),
                "importance_scores": shap_explanations.get('importance_scores', {})
            },
            "counterfactuals": {
                "scenarios": counterfactuals,
                "interpretation": _interpret_counterfactuals(counterfactuals)
            },
            "temporal_analysis": {
                "score_trend": temporal_explanation.get('trend', 'stable'),
                "key_changes": temporal_explanation.get('key_changes', []),
                "trend_drivers": temporal_explanation.get('drivers', [])
            },
            "plain_text": plain_text_summary,
            "peer_comparison": peer_comparison,
            "confidence_metrics": {
                "prediction_confidence": round(prediction_result['confidence'][0], 3),
                "explanation_stability": shap_explanations.get('stability_score', 0.0),
                "data_quality_score": current_data.get('quality_score', 1.0)
            },
            "generated_at": datetime.now().isoformat(),
            "model_version": model_service.get_model_version()
        }
        
        logger.info(f"Generated explanations for {ticker} with score {current_score}")
        return jsonify(response)
        
    except APIError as e:
        return handle_api_error(e)
    except Exception as e:
        logger.error(f"Error generating explanations for {ticker}: {str(e)}")
        return handle_api_error(APIError("Error generating explanations", status_code=500))

def _interpret_counterfactuals(counterfactuals: List[Dict]) -> str:
    """Generate interpretation of counterfactual scenarios"""
    if not counterfactuals:
        return "No alternative scenarios available."
    
    interpretations = []
    for cf in counterfactuals:
        target_score = cf.get('target_score', 0)
        changes = cf.get('required_changes', {})
        
        if changes:
            key_change = max(changes.items(), key=lambda x: abs(x[1]))
            feature, change_amount = key_change
            
            direction = "increase" if change_amount > 0 else "decrease"
            interpretations.append(f"To reach score {target_score}, primarily need to {direction} {feature}")
    
    return "; ".join(interpretations[:2])  # Limit to 2 most relevant

# =====================================================
# PEER COMPARISON ENDPOINT
# =====================================================

@realtime_bp.route('/peers/<ticker>', methods=['GET'])
@limiter.limit("20 per hour")
@monitor_performance
@cache_response(ttl=900, key_prefix="peers")  # 15-minute cache
def get_peer_comparison(ticker: str):
    """Get peer comparison data for a ticker"""
    try:
        if not validate_ticker(ticker):
            raise APIError("Invalid ticker symbol", status_code=400)
        
        ticker = ticker.upper().strip()
        
        logger.info(f"Generating peer comparison for {ticker}")
        
        # Get company information
        company_info = data_collector.get_company_info(ticker)
        if not company_info:
            raise APIError(f"Company information not found for {ticker}", status_code=404)
        
        sector = company_info.get('sector', 'Unknown')
        market_cap_category = _get_market_cap_category(company_info.get('market_cap', 0))
        
        # Get peer companies
        peer_tickers = data_collector.get_sector_peers(
            ticker=ticker,
            sector=sector,
            market_cap_category=market_cap_category,
            limit=10
        )
        
        # Get scores for all companies
        all_scores = {}
        target_score = None
        
        for peer_ticker in [ticker] + peer_tickers:
            try:
                peer_data = data_collector.get_latest_features(peer_ticker)
                if peer_data:
                    prediction = model_service.predict_with_uncertainty([peer_data])
                    score = prediction['predictions'][0]
                    confidence = prediction['confidence'][0]
                    
                    all_scores[peer_ticker] = {
                        'score': float(score),
                        'confidence': float(confidence),
                        'market_cap': peer_data.get('market_cap', 0),
                        'company_name': peer_data.get('company_name', peer_ticker)
                    }
                    
                    if peer_ticker == ticker:
                        target_score = float(score)
            except Exception as e:
                logger.warning(f"Error getting score for peer {peer_ticker}: {e}")
                continue
        
        if target_score is None:
            raise APIError(f"Could not get score for {ticker}", status_code=404)
        
        # Calculate sector statistics
        peer_scores = [data['score'] for t, data in all_scores.items() if t != ticker]
        
        if peer_scores:
            sector_avg = sum(peer_scores) / len(peer_scores)
            sector_median = sorted(peer_scores)[len(peer_scores)//2]
            sector_std = (sum((x - sector_avg) ** 2 for x in peer_scores) / len(peer_scores)) ** 0.5
        else:
            sector_avg = target_score
            sector_median = target_score
            sector_std = 0.0
        
        # Calculate percentile ranking
        all_peer_scores = peer_scores + [target_score]
        all_peer_scores.sort()
        percentile_rank = (all_peer_scores.index(target_score) + 1) / len(all_peer_scores) * 100
        
        # Generate peer rankings
        ranked_peers = sorted(
            [(t, data) for t, data in all_scores.items()],
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Create response
        response = {
            "ticker": ticker,
            "current_score": round(target_score, 2),
            "sector": sector,
            "market_cap_category": market_cap_category,
            "sector_statistics": {
                "average": round(sector_avg, 2),
                "median": round(sector_median, 2),
                "standard_deviation": round(sector_std, 2),
                "peer_count": len(peer_scores)
            },
            "relative_position": {
                "percentile_rank": round(percentile_rank, 1),
                "vs_sector_avg": round(target_score - sector_avg, 2),
                "vs_sector_median": round(target_score - sector_median, 2),
                "rank": next(i for i, (t, _) in enumerate(ranked_peers, 1) if t == ticker),
                "total_companies": len(ranked_peers)
            },
            "peer_rankings": [
                {
                    "ticker": t,
                    "company_name": data['company_name'],
                    "score": round(data['score'], 2),
                    "confidence": round(data['confidence'], 3),
                    "market_cap": data['market_cap'],
                    "rank": i
                }
                for i, (t, data) in enumerate(ranked_peers, 1)
            ],
            "analysis": {
                "relative_strength": _analyze_relative_strength(target_score, sector_avg, percentile_rank),
                "key_differentiators": _identify_key_differentiators(ticker, peer_tickers),
                "risk_assessment": _assess_peer_risk(percentile_rank, sector_std)
            },
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(f"Generated peer comparison for {ticker} - Rank {response['relative_position']['rank']} of {len(ranked_peers)}")
        return jsonify(response)
        
    except APIError as e:
        return handle_api_error(e)
    except Exception as e:
        logger.error(f"Error generating peer comparison for {ticker}: {str(e)}")
        return handle_api_error(APIError("Error generating peer comparison", status_code=500))

def _get_market_cap_category(market_cap: float) -> str:
    """Categorize market cap for peer selection"""
    if market_cap > 200_000_000_000:  # $200B+
        return "mega_cap"
    elif market_cap > 10_000_000_000:  # $10B+
        return "large_cap"
    elif market_cap > 2_000_000_000:   # $2B+
        return "mid_cap"
    else:
        return "small_cap"

def _analyze_relative_strength(score: float, sector_avg: float, percentile: float) -> str:
    """Analyze relative strength vs peers"""
    if percentile >= 75:
        return f"Strong performer - significantly above sector average by {score - sector_avg:.1f} points"
    elif percentile >= 50:
        return f"Above average performer - {score - sector_avg:.1f} points above sector average"
    elif percentile >= 25:
        return f"Below average performer - {abs(score - sector_avg):.1f} points below sector average"
    else:
        return f"Weak performer - significantly below sector average by {abs(score - sector_avg):.1f} points"

def _identify_key_differentiators(ticker: str, peers: List[str]) -> List[str]:
    """Identify key differentiating factors vs peers"""
    # This would analyze feature differences between target company and peers
    # Simplified implementation for now
    return [
        "Financial leverage ratios",
        "Profitability metrics",
        "Market position strength"
    ]

def _assess_peer_risk(percentile: float, sector_std: float) -> str:
    """Assess risk relative to peer group"""
    if sector_std > 10:
        volatility = "high"
    elif sector_std > 5:
        volatility = "moderate"
    else:
        volatility = "low"
    
    if percentile < 25:
        return f"Higher risk relative to peers in {volatility} volatility sector"
    elif percentile > 75:
        return f"Lower risk relative to peers in {volatility} volatility sector"
    else:
        return f"Average risk profile relative to peers in {volatility} volatility sector"

# =====================================================
# WEBSOCKET SUPPORT
# =====================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    logger.info(f"Client connected: {client_id}")
    
    # Send initial connection confirmation
    emit('connection_status', {
        'status': 'connected',
        'client_id': client_id,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    logger.info(f"Client disconnected: {client_id}")

@socketio.on('subscribe_scores')
def handle_subscribe_scores(data):
    """Subscribe to real-time score updates for specific tickers"""
    try:
        tickers = data.get('tickers', [])
        client_id = request.sid
        
        for ticker in tickers:
            if validate_ticker(ticker):
                room_name = f"scores_{ticker.upper()}"
                join_room(room_name)
                logger.info(f"Client {client_id} subscribed to scores for {ticker}")
        
        emit('subscription_status', {
            'status': 'subscribed',
            'tickers': tickers,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error subscribing to scores: {e}")
        emit('error', {'message': 'Subscription failed'})

@socketio.on('unsubscribe_scores')
def handle_unsubscribe_scores(data):
    """Unsubscribe from score updates"""
    try:
        tickers = data.get('tickers', [])
        client_id = request.sid
        
        for ticker in tickers:
            if validate_ticker(ticker):
                room_name = f"scores_{ticker.upper()}"
                leave_room(room_name)
                logger.info(f"Client {client_id} unsubscribed from scores for {ticker}")
        
        emit('subscription_status', {
            'status': 'unsubscribed',
            'tickers': tickers,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error unsubscribing from scores: {e}")
        emit('error', {'message': 'Unsubscription failed'})

@socketio.on('subscribe_alerts')
def handle_subscribe_alerts(data):
    """Subscribe to real-time alert notifications"""
    try:
        user_id = data.get('user_id', 'default_user')
        client_id = request.sid
        
        room_name = f"alerts_{user_id}"
        join_room(room_name)
        
        logger.info(f"Client {client_id} subscribed to alerts for user {user_id}")
        
        emit('subscription_status', {
            'status': 'subscribed',
            'type': 'alerts',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error subscribing to alerts: {e}")
        emit('error', {'message': 'Alert subscription failed'})

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def _get_user_watchlist(user_id: str) -> List[str]:
    """Get user's watchlist from cache/database"""
    if redis_client:
        watchlist = redis_client.get(f"watchlist:{user_id}")
        if watchlist:
            return json.loads(watchlist)
    
    # Default watchlist for demo
    return ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

def _get_ticker_alerts(ticker: str) -> List[Dict]:
    """Get all alert rules for a ticker"""
    if redis_client:
        alerts = redis_client.get(f"alerts:{ticker}")
        if alerts:
            return json.loads(alerts)
    return []

def _check_alert_conditions(ticker: str) -> Dict:
    """Check current conditions for alert evaluation"""
    try:
        # Get current data
        current_data = data_collector.get_latest_features(ticker)
        if not current_data:
            return {}
        
        # Get current score
        prediction = model_service.predict_with_uncertainty([current_data])
        current_score = prediction['predictions'][0]
        
        # Get additional metrics
        market_data = data_collector.get_market_context(ticker)
        
        return {
            'score': float(current_score),
            'volatility': market_data.get('volatility_30d', 0.0),
            'price_change': market_data.get('price_change_24h', 0.0),
            'volume': market_data.get('volume', 0),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking alert conditions for {ticker}: {e}")
        return {}

def _evaluate_alert_condition(alert: Dict, conditions: Dict) -> bool:
    """Evaluate if alert condition is met"""
    try:
        condition_field = alert.get('condition_field', 'score')
        threshold = alert['threshold']
        condition_type = alert['condition']
        current_value = conditions.get(condition_field)
        
        if current_value is None:
            return False
        
        if condition_type == 'less_than':
            return current_value < threshold
        elif condition_type == 'greater_than':
            return current_value > threshold
        elif condition_type == 'equals':
            return abs(current_value - threshold) < 0.01
        
        return False
        
    except Exception as e:
        logger.error(f"Error evaluating alert condition: {e}")
        return False

def _store_alert_rule(alert_rule: Dict):
    """Store alert rule in cache/database"""
    if redis_client:
        ticker = alert_rule['ticker']
        alert_id = alert_rule['id']
        
        # Store individual alert
        redis_client.set(f"alert:{alert_id}", json.dumps(alert_rule), ex=86400*30)  # 30 days
        
        # Add to ticker alerts list
        ticker_alerts = _get_ticker_alerts(ticker)
        ticker_alerts.append(alert_rule)
        redis_client.set(f"alerts:{ticker}", json.dumps(ticker_alerts), ex=86400*30)

def _delete_alert_rule(alert_id: str) -> bool:
    """Delete alert rule from cache/database"""
    if redis_client:
        # Get alert details
        alert_data = redis_client.get(f"alert:{alert_id}")
        if not alert_data:
            return False
        
        alert = json.loads(alert_data)
        ticker = alert['ticker']
        
        # Remove individual alert
        redis_client.delete(f"alert:{alert_id}")
        
        # Remove from ticker alerts list
        ticker_alerts = _get_ticker_alerts(ticker)
        ticker_alerts = [a for a in ticker_alerts if a['id'] != alert_id]
        redis_client.set(f"alerts:{ticker}", json.dumps(ticker_alerts), ex=86400*30)
        
        return True
    
    return False

# =====================================================
# BACKGROUND TASKS FOR REAL-TIME UPDATES
# =====================================================

def start_background_tasks():
    """Start background tasks for real-time updates"""
    executor = ThreadPoolExecutor(max_workers=3)
    
    # Start score monitoring task
    executor.submit(score_monitoring_task)
    
    # Start alert checking task
    executor.submit(alert_checking_task)
    
    logger.info("Background tasks started for real-time API")

def score_monitoring_task():
    """Background task to monitor and broadcast score changes"""
    while True:
        try:
            # Get list of actively monitored tickers
            active_tickers = _get_active_tickers()
            
            for ticker in active_tickers:
                try:
                    # Check if score has changed significantly
                    current_score = _get_current_score(ticker)
                    if current_score is not None:
                        last_score = _get_last_broadcasted_score(ticker)
                        
                        if last_score is None or abs(current_score - last_score) >= 1.0:
                            # Broadcast score update
                            socketio.emit('score_update', {
                                'ticker': ticker,
                                'score': current_score,
                                'change': current_score - (last_score or current_score),
                                'timestamp': datetime.now().isoformat()
                            }, room=f"scores_{ticker}")
                            
                            # Update last broadcasted score
                            _update_last_broadcasted_score(ticker, current_score)
                            
                except Exception as e:
                    logger.error(f"Error monitoring score for {ticker}: {e}")
                    continue
            
            # Wait 30 seconds before next check
            time.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in score monitoring task: {e}")
            time.sleep(60)  # Wait longer on error

def alert_checking_task():
    """Background task to check and trigger alerts"""
    while True:
        try:
            # Get all active alerts
            all_alerts = _get_all_active_alerts()
            
            for alert in all_alerts:
                try:
                    ticker = alert['ticker']
                    
                    # Check alert condition
                    conditions = _check_alert_conditions(ticker)
                    if _evaluate_alert_condition(alert, conditions):
                        # Trigger alert
                        alert_data = {
                            'alert_id': alert['id'],
                            'ticker': ticker,
                            'type': alert['alert_type'],
                            'message': alert.get('message_template', '').format(ticker=ticker),
                            'severity': alert['severity'],
                            'current_value': conditions.get(alert['condition_field']),
                            'threshold': alert['threshold'],
                            'triggered_at': datetime.now().isoformat()
                        }
                        
                        # Broadcast alert
                        user_id = alert['user_id']
                        socketio.emit('alert_triggered', alert_data, room=f"alerts_{user_id}")
                        
                        # Log alert
                        logger.info(f"Alert triggered: {alert['id']} for {ticker}")
                        
                except Exception as e:
                    logger.error(f"Error checking alert {alert.get('id', 'unknown')}: {e}")
                    continue
            
            # Wait 60 seconds before next check
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in alert checking task: {e}")
            time.sleep(120)  # Wait longer on error

def _get_active_tickers() -> List[str]:
    """Get list of tickers being actively monitored"""
    if redis_client:
        # Get tickers that have active subscriptions
        active_rooms = []
        # In a real implementation, you'd query Socket.IO for active rooms
        # For now, return a default list
        return ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    return []

def _get_current_score(ticker: str) -> Optional[float]:
    """Get current score for a ticker"""
    try:
        current_data = data_collector.get_latest_features(ticker)
        if current_data:
            prediction = model_service.predict_with_uncertainty([current_data])
            return float(prediction['predictions'][0])
    except Exception:
        pass
    return None

def _get_last_broadcasted_score(ticker: str) -> Optional[float]:
    """Get last broadcasted score for a ticker"""
    if redis_client:
        score = redis_client.get(f"last_broadcast_score:{ticker}")
        return float(score) if score else None
    return None

def _update_last_broadcasted_score(ticker: str, score: float):
    """Update last broadcasted score for a ticker"""
    if redis_client:
        redis_client.set(f"last_broadcast_score:{ticker}", score, ex=3600)  # 1 hour TTL

def _get_all_active_alerts() -> List[Dict]:
    """Get all active alert rules"""
    if redis_client:
        # In a real implementation, you'd scan for all alert keys
        # For now, return alerts for active tickers
        all_alerts = []
        active_tickers = _get_active_tickers()
        
        for ticker in active_tickers:
            ticker_alerts = _get_ticker_alerts(ticker)
            all_alerts.extend([a for a in ticker_alerts if a.get('enabled', True)])
        
        return all_alerts
    return []

# Initialize background tasks when module loads
# start_background_tasks()

# Export socketio for app initialization
__all__ = ['realtime_bp', 'socketio']
