# API route for monitoring
# File: /api/routes/monitoring.py

import os
import time
import psutil
import smtplib
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import sqlite3

from flask import Blueprint, jsonify, request
import prometheus_client
from prometheus_client import CollectorRegistry, generate_latest, Gauge, Counter, Histogram
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create monitoring blueprint
monitoring_bp = Blueprint('monitoring', __name__)

@dataclass
class HealthStatus:
    """Data class for health status tracking"""
    service: str
    status: str  # 'healthy', 'warning', 'critical'
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None

@dataclass
class AlertRule:
    """Data class for alert configuration"""
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str  # 'critical', 'warning', 'info'
    email_recipients: List[str]
    last_triggered: Optional[datetime] = None

class SystemMonitor:
    """
    Comprehensive system monitoring and health check manager
    """
    
    def __init__(self):
        """Initialize system monitor with Prometheus metrics and alerting"""
        
        # Prometheus registry
        self.registry = CollectorRegistry(auto_describe=True)
        
        # System metrics
        self.system_cpu_gauge = Gauge('system_cpu_percent', 'System CPU usage percentage', registry=self.registry)
        self.system_memory_gauge = Gauge('system_memory_percent', 'System memory usage percentage', registry=self.registry)
        self.system_disk_gauge = Gauge('system_disk_percent', 'System disk usage percentage', registry=self.registry)
        
        # API metrics
        self.api_requests_counter = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'], registry=self.registry)
        self.api_latency_histogram = Histogram('api_request_duration_seconds', 'API request latency', ['endpoint'], registry=self.registry)
        self.active_connections_gauge = Gauge('active_connections', 'Number of active connections', registry=self.registry)
        
        # Data quality metrics
        self.data_freshness_gauge = Gauge('data_freshness_seconds', 'Data freshness in seconds', ['source'], registry=self.registry)
        self.data_missing_percentage_gauge = Gauge('data_missing_percentage', 'Percentage of missing data', ['feature'], registry=self.registry)
        self.api_connection_gauge = Gauge('external_api_status', 'External API connection status', ['api'], registry=self.registry)
        
        # Model performance metrics
        self.model_accuracy_gauge = Gauge('model_accuracy', 'Model accuracy score', ['model'], registry=self.registry)
        self.model_drift_gauge = Gauge('model_drift_score', 'Model drift detection score', ['model'], registry=self.registry)
        self.prediction_latency_histogram = Histogram('model_prediction_duration_seconds', 'Model prediction latency', ['model'], registry=self.registry)
        
        # Cost metrics
        self.api_cost_gauge = Gauge('external_api_cost_usd', 'External API costs in USD', ['api'], registry=self.registry)
        self.infrastructure_cost_gauge = Gauge('infrastructure_cost_usd', 'Infrastructure costs in USD', ['component'], registry=self.registry)
        
        # Health status storage
        self.health_status = {}
        self.alert_rules = []
        self.metrics_cache = {}
        
        # Email configuration
        self.smtp_config = {
            'server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('SMTP_USERNAME'),
            'password': os.getenv('SMTP_PASSWORD'),
            'from_email': os.getenv('ALERT_FROM_EMAIL')
        }
        
        # Initialize alert rules
        self._initialize_alert_rules()
        
        # Start background monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("SystemMonitor initialized with Prometheus metrics and alerting")
    
    def _initialize_alert_rules(self):
        """Initialize default alert rules"""
        self.alert_rules = [
            AlertRule('system_cpu_percent', 80.0, 'gt', 'warning', ['admin@company.com']),
            AlertRule('system_memory_percent', 85.0, 'gt', 'critical', ['admin@company.com']),
            AlertRule('model_accuracy', 0.8, 'lt', 'critical', ['ml-team@company.com']),
            AlertRule('model_drift_score', 0.3, 'gt', 'warning', ['ml-team@company.com']),
            AlertRule('api_latency_p95', 2.0, 'gt', 'warning', ['dev-team@company.com']),
            AlertRule('external_api_cost_usd', 100.0, 'gt', 'warning', ['finance@company.com'])
        ]
    
    def _background_monitor(self):
        """Background monitoring thread for periodic health checks"""
        while self.monitoring_active:
            try:
                self._update_system_metrics()
                self._check_data_quality()
                self._check_model_performance()
                self._check_external_apis()
                self._evaluate_alert_rules()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                time.sleep(60)
    
    def _update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_gauge.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_gauge.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_disk_gauge.set(disk_percent)
            
            # Update cache
            self.metrics_cache['system'] = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk_percent,
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _check_data_quality(self):
        """Check data quality metrics"""
        try:
            # Simulate data quality checks (replace with real implementation)
            data_sources = ['yahoo_finance', 'news_api', 'edgar_api']
            
            for source in data_sources:
                # Check data freshness (mock implementation)
                freshness_seconds = self._get_data_freshness(source)
                self.data_freshness_gauge.labels(source=source).set(freshness_seconds)
                
                # Check API connection
                api_status = self._check_api_connection(source)
                self.api_connection_gauge.labels(api=source).set(1 if api_status else 0)
            
            # Update cache
            self.metrics_cache['data_quality'] = {
                'freshness': {source: self._get_data_freshness(source) for source in data_sources},
                'api_status': {source: self._check_api_connection(source) for source in data_sources},
                'missing_data_percentage': 0.05,  # Mock value
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
    
    def _check_model_performance(self):
        """Check model performance metrics"""
        try:
            # Mock model performance checks (replace with real implementation)
            models = ['lightgbm', 'xgboost', 'ensemble']
            
            for model in models:
                # Model accuracy (from recent predictions)
                accuracy = self._get_model_accuracy(model)
                self.model_accuracy_gauge.labels(model=model).set(accuracy)
                
                # Model drift detection
                drift_score = self._get_model_drift_score(model)
                self.model_drift_gauge.labels(model=model).set(drift_score)
            
            # Update cache
            self.metrics_cache['model_performance'] = {
                'accuracy': {model: self._get_model_accuracy(model) for model in models},
                'drift_scores': {model: self._get_model_drift_score(model) for model in models},
                'feature_stability': 0.92,  # Mock value
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
    
    def _check_external_apis(self):
        """Check external API connections and costs"""
        try:
            # API cost tracking (mock implementation)
            api_costs = {
                'openai': self._get_openai_cost(),
                'news_api': 0.0,  # Free tier
                'yahoo_finance': 0.0  # Free
            }
            
            for api, cost in api_costs.items():
                self.api_cost_gauge.labels(api=api).set(cost)
            
            # Infrastructure costs
            infrastructure_costs = {
                'compute': 25.50,
                'storage': 5.25,
                'bandwidth': 3.75
            }
            
            for component, cost in infrastructure_costs.items():
                self.infrastructure_cost_gauge.labels(component=component).set(cost)
            
            # Update cache
            self.metrics_cache['costs'] = {
                'api_costs': api_costs,
                'infrastructure_costs': infrastructure_costs,
                'total_monthly_cost': sum(api_costs.values()) + sum(infrastructure_costs.values()),
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking external APIs: {e}")
    
    def _evaluate_alert_rules(self):
        """Evaluate alert rules and send notifications"""
        try:
            for rule in self.alert_rules:
                current_value = self._get_metric_value(rule.metric_name)
                
                if current_value is None:
                    continue
                
                # Check if alert should trigger
                should_alert = False
                
                if rule.comparison == 'gt' and current_value > rule.threshold:
                    should_alert = True
                elif rule.comparison == 'lt' and current_value < rule.threshold:
                    should_alert = True
                elif rule.comparison == 'eq' and current_value == rule.threshold:
                    should_alert = True
                
                # Send alert if needed (with cooldown period)
                if should_alert:
                    cooldown_minutes = 30 if rule.severity == 'critical' else 60
                    
                    if (rule.last_triggered is None or 
                        datetime.now() - rule.last_triggered > timedelta(minutes=cooldown_minutes)):
                        
                        self._send_alert(rule, current_value)
                        rule.last_triggered = datetime.now()
                        
        except Exception as e:
            logger.error(f"Error evaluating alert rules: {e}")
    
    def _send_alert(self, rule: AlertRule, current_value: float):
        """Send alert notification via email"""
        try:
            if not all([self.smtp_config['username'], self.smtp_config['password'], self.smtp_config['from_email']]):
                logger.warning("SMTP configuration incomplete, cannot send email alerts")
                return
            
            subject = f"[{rule.severity.upper()}] Credit Intelligence Platform Alert"
            
            body = f"""
            Alert: {rule.metric_name}
            
            Current Value: {current_value}
            Threshold: {rule.threshold}
            Severity: {rule.severity}
            Timestamp: {datetime.now().isoformat()}
            
            Please investigate immediately.
            
            --
            Credit Intelligence Platform Monitoring System
            """
            
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = ', '.join(rule.email_recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                server.starttls()
                server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
            
            logger.info(f"Alert sent for {rule.metric_name}: {current_value}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric"""
        # Map metric names to actual values from cache
        metric_mappings = {
            'system_cpu_percent': lambda: self.metrics_cache.get('system', {}).get('cpu_percent'),
            'system_memory_percent': lambda: self.metrics_cache.get('system', {}).get('memory_percent'),
            'model_accuracy': lambda: min(self.metrics_cache.get('model_performance', {}).get('accuracy', {}).values()),
            'model_drift_score': lambda: max(self.metrics_cache.get('model_performance', {}).get('drift_scores', {}).values()),
            'external_api_cost_usd': lambda: sum(self.metrics_cache.get('costs', {}).get('api_costs', {}).values())
        }
        
        if metric_name in metric_mappings:
            try:
                return metric_mappings[metric_name]()
            except:
                return None
        
        return None
    
    # Helper methods for data collection (implement with real data sources)
    def _get_data_freshness(self, source: str) -> float:
        """Get data freshness for a source in seconds"""
        # Mock implementation - replace with real data age calculation
        freshness_map = {'yahoo_finance': 300, 'news_api': 600, 'edgar_api': 3600}
        return freshness_map.get(source, 1800)
    
    def _check_api_connection(self, source: str) -> bool:
        """Check if external API is accessible"""
        # Mock implementation - replace with real API health checks
        try:
            # You would implement actual API calls here
            return True
        except:
            return False
    
    def _get_model_accuracy(self, model: str) -> float:
        """Get recent model accuracy"""
        # Mock implementation - replace with real model performance tracking
        accuracy_map = {'lightgbm': 0.92, 'xgboost': 0.89, 'ensemble': 0.94}
        return accuracy_map.get(model, 0.85)
    
    def _get_model_drift_score(self, model: str) -> float:
        """Get model drift detection score"""
        # Mock implementation - replace with real drift detection
        drift_map = {'lightgbm': 0.15, 'xgboost': 0.12, 'ensemble': 0.08}
        return drift_map.get(model, 0.20)
    
    def _get_openai_cost(self) -> float:
        """Get OpenAI API usage cost"""
        # Mock implementation - replace with real cost tracking
        return 45.75

# Initialize global monitor instance
monitor = SystemMonitor()

# Health Check Endpoints

@monitoring_bp.route('/health/status', methods=['GET'])
def health_status():
    """
    Overall system health status endpoint
    Returns comprehensive system health information
    """
    try:
        # Get current system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Determine overall health status
        health_score = 100
        issues = []
        
        if cpu_percent > 80:
            health_score -= 20
            issues.append(f"High CPU usage: {cpu_percent}%")
        
        if memory.percent > 85:
            health_score -= 25
            issues.append(f"High memory usage: {memory.percent}%")
        
        if (disk.used / disk.total) * 100 > 90:
            health_score -= 15
            issues.append("High disk usage")
        
        # Check external dependencies
        external_services = {
            'database': True,  # Mock - implement real DB health check
            'redis_cache': True,  # Mock - implement real Redis check
            'model_service': True  # Mock - implement real model health check
        }
        
        if not all(external_services.values()):
            health_score -= 30
            issues.extend([f"{service} unavailable" for service, status in external_services.items() if not status])
        
        overall_status = 'healthy' if health_score >= 80 else 'warning' if health_score >= 60 else 'critical'
        
        response = {
            'status': overall_status,
            'health_score': health_score,
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100,
                'uptime_seconds': time.time() - psutil.boot_time()
            },
            'external_services': external_services,
            'issues': issues,
            'response_time_ms': 250  # Mock benchmark
        }
        
        return jsonify(response), 200 if overall_status == 'healthy' else 503
        
    except Exception as e:
        logger.error(f"Error in health status check: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@monitoring_bp.route('/health/data-quality', methods=['GET'])
def health_data_quality():
    """
    Data quality health check endpoint
    Returns data freshness, completeness, and API connectivity status
    """
    try:
        data_quality_metrics = monitor.metrics_cache.get('data_quality', {})
        
        # Calculate overall data quality score
        quality_score = 100
        issues = []
        
        # Check data freshness
        freshness = data_quality_metrics.get('freshness', {})
        for source, age_seconds in freshness.items():
            if age_seconds > 3600:  # More than 1 hour old
                quality_score -= 15
                issues.append(f"{source} data is {age_seconds/3600:.1f} hours old")
        
        # Check API connectivity
        api_status = data_quality_metrics.get('api_status', {})
        failed_apis = [api for api, status in api_status.items() if not status]
        if failed_apis:
            quality_score -= 25 * len(failed_apis)
            issues.extend([f"{api} API connection failed" for api in failed_apis])
        
        # Check missing data percentage
        missing_percentage = data_quality_metrics.get('missing_data_percentage', 0)
        if missing_percentage > 0.1:  # More than 10% missing
            quality_score -= 20
            issues.append(f"High missing data: {missing_percentage*100:.1f}%")
        
        overall_status = 'healthy' if quality_score >= 80 else 'warning' if quality_score >= 60 else 'critical'
        
        response = {
            'status': overall_status,
            'quality_score': quality_score,
            'timestamp': datetime.now().isoformat(),
            'data_sources': {
                'freshness_seconds': freshness,
                'api_connectivity': api_status,
                'missing_data_percentage': missing_percentage
            },
            'issues': issues,
            'last_quality_check': data_quality_metrics.get('last_update')
        }
        
        return jsonify(response), 200 if overall_status == 'healthy' else 503
        
    except Exception as e:
        logger.error(f"Error in data quality check: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@monitoring_bp.route('/health/model-performance', methods=['GET'])
def health_model_performance():
    """
    Model performance health check endpoint
    Returns model accuracy, drift detection, and feature stability
    """
    try:
        model_metrics = monitor.metrics_cache.get('model_performance', {})
        
        # Calculate overall model health score
        health_score = 100
        issues = []
        
        # Check model accuracy
        accuracy_scores = model_metrics.get('accuracy', {})
        min_accuracy = min(accuracy_scores.values()) if accuracy_scores else 0.85
        
        if min_accuracy < 0.8:
            health_score -= 30
            issues.append(f"Low model accuracy: {min_accuracy:.3f}")
        elif min_accuracy < 0.85:
            health_score -= 15
            issues.append(f"Model accuracy below target: {min_accuracy:.3f}")
        
        # Check model drift
        drift_scores = model_metrics.get('drift_scores', {})
        max_drift = max(drift_scores.values()) if drift_scores else 0.1
        
        if max_drift > 0.3:
            health_score -= 25
            issues.append(f"High model drift detected: {max_drift:.3f}")
        elif max_drift > 0.2:
            health_score -= 10
            issues.append(f"Moderate model drift detected: {max_drift:.3f}")
        
        # Check feature stability
        feature_stability = model_metrics.get('feature_stability', 0.92)
        if feature_stability < 0.85:
            health_score -= 20
            issues.append(f"Low feature stability: {feature_stability:.3f}")
        
        overall_status = 'healthy' if health_score >= 80 else 'warning' if health_score >= 60 else 'critical'
        
        response = {
            'status': overall_status,
            'health_score': health_score,
            'timestamp': datetime.now().isoformat(),
            'model_metrics': {
                'accuracy_by_model': accuracy_scores,
                'drift_scores_by_model': drift_scores,
                'feature_stability': feature_stability,
                'min_accuracy': min_accuracy,
                'max_drift': max_drift
            },
            'performance_thresholds': {
                'min_accuracy_threshold': 0.8,
                'max_drift_threshold': 0.3,
                'min_feature_stability': 0.85
            },
            'issues': issues,
            'last_performance_check': model_metrics.get('last_update')
        }
        
        return jsonify(response), 200 if overall_status == 'healthy' else 503
        
    except Exception as e:
        logger.error(f"Error in model performance check: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Metrics Endpoints

@monitoring_bp.route('/metrics/usage', methods=['GET'])
def metrics_usage():
    """
    API usage and user activity metrics endpoint
    """
    try:
        # Mock usage data - replace with real usage tracking
        usage_data = {
            'api_calls': {
                'predict': {'today': 1250, 'total': 45678},
                'explain': {'today': 980, 'total': 32456},
                'realtime': {'today': 560, 'total': 15678}
            },
            'active_users': {
                'current': 25,
                'today': 156,
                'this_week': 892
            },
            'performance_metrics': {
                'avg_response_time_ms': 345,
                'p95_response_time_ms': 890,
                'error_rate_percent': 0.5,
                'uptime_percent': 99.8
            },
            'geographic_distribution': {
                'north_america': 65,
                'europe': 25,
                'asia': 10
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(usage_data), 200
        
    except Exception as e:
        logger.error(f"Error in usage metrics: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@monitoring_bp.route('/metrics/costs', methods=['GET'])
def metrics_costs():
    """
    Cost monitoring and budget tracking endpoint
    """
    try:
        cost_metrics = monitor.metrics_cache.get('costs', {})
        
        # Calculate projections
        daily_api_cost = sum(cost_metrics.get('api_costs', {}).values()) / 30  # Approximate daily cost
        monthly_projection = daily_api_cost * 30
        
        infrastructure_total = sum(cost_metrics.get('infrastructure_costs', {}).values())
        
        cost_data = {
            'api_costs_usd': cost_metrics.get('api_costs', {}),
            'infrastructure_costs_usd': cost_metrics.get('infrastructure_costs', {}),
            'total_monthly_cost_usd': cost_metrics.get('total_monthly_cost', 0),
            'projections': {
                'daily_api_cost_usd': daily_api_cost,
                'monthly_api_projection_usd': monthly_projection,
                'annual_projection_usd': (monthly_projection + infrastructure_total) * 12
            },
            'budget_status': {
                'monthly_budget_usd': 200.0,
                'budget_used_percent': (cost_metrics.get('total_monthly_cost', 0) / 200.0) * 100,
                'remaining_budget_usd': max(0, 200.0 - cost_metrics.get('total_monthly_cost', 0))
            },
            'cost_breakdown': {
                'api_percentage': (sum(cost_metrics.get('api_costs', {}).values()) / max(cost_metrics.get('total_monthly_cost', 1), 1)) * 100,
                'infrastructure_percentage': (infrastructure_total / max(cost_metrics.get('total_monthly_cost', 1), 1)) * 100
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(cost_data), 200
        
    except Exception as e:
        logger.error(f"Error in cost metrics: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Prometheus Metrics Export

@monitoring_bp.route('/metrics/prometheus', methods=['GET'])
def metrics_prometheus():
    """
    Prometheus-compatible metrics export endpoint
    """
    try:
        # Update all metrics before export
        monitor._update_system_metrics()
        
        # Generate Prometheus format
        metrics_data = generate_latest(monitor.registry)
        
        return metrics_data, 200, {
            'Content-Type': prometheus_client.CONTENT_TYPE_LATEST
        }
        
    except Exception as e:
        logger.error(f"Error exporting Prometheus metrics: {e}")
        return f"# Error exporting metrics: {str(e)}", 500

# Alerting Configuration

@monitoring_bp.route('/alerts/rules', methods=['GET'])
def get_alert_rules():
    """Get current alert rules configuration"""
    try:
        rules_data = [asdict(rule) for rule in monitor.alert_rules]
        return jsonify({
            'alert_rules': rules_data,
            'total_rules': len(monitor.alert_rules),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting alert rules: {e}")
        return jsonify({'error': str(e)}), 500

@monitoring_bp.route('/alerts/rules', methods=['POST'])
def add_alert_rule():
    """Add a new alert rule"""
    try:
        data = request.get_json()
        
        new_rule = AlertRule(
            metric_name=data['metric_name'],
            threshold=float(data['threshold']),
            comparison=data['comparison'],
            severity=data['severity'],
            email_recipients=data['email_recipients']
        )
        
        monitor.alert_rules.append(new_rule)
        
        return jsonify({
            'message': 'Alert rule added successfully',
            'rule': asdict(new_rule)
        }), 201
        
    except Exception as e:
        logger.error(f"Error adding alert rule: {e}")
        return jsonify({'error': str(e)}), 400

@monitoring_bp.route('/alerts/test', methods=['POST'])
def test_alert():
    """Test alert notification system"""
    try:
        data = request.get_json()
        email = data.get('email', 'test@example.com')
        
        # Create test alert rule
        test_rule = AlertRule(
            metric_name='test_metric',
            threshold=0.0,
            comparison='gt',
            severity='info',
            email_recipients=[email]
        )
        
        monitor._send_alert(test_rule, 1.0)
        
        return jsonify({
            'message': 'Test alert sent successfully',
            'recipient': email
        }), 200
        
    except Exception as e:
        logger.error(f"Error sending test alert: {e}")
        return jsonify({'error': str(e)}), 500

# Utility endpoints

@monitoring_bp.route('/system/info', methods=['GET'])
def system_info():
    """Get detailed system information"""
    try:
        import platform
        
        system_info = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'python': {
                'version': platform.python_version(),
                'implementation': platform.python_implementation()
            },
            'resources': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3)
            },
            'network': {
                'hostname': platform.node()
            },
            'application': {
                'name': 'Credit Intelligence Platform',
                'version': '1.0.0',
                'environment': os.getenv('ENVIRONMENT', 'development')
            }
        }
        
        return jsonify(system_info), 200
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize monitoring when module is imported
logger.info("Monitoring system initialized and ready")

# Example usage for integration with main Flask app:
"""
from api.routes.monitoring import monitoring_bp

app = Flask(__name__)
app.register_blueprint(monitoring_bp, url_prefix='/api')

# The monitoring system will automatically start background health checks
# and expose all endpoints for system monitoring and alerting
"""
