# Online learning model implementation
# File: /api/models/online_learner.py

"""
Implement the OnlineLearner class for a real-time credit scoring platform.

Features to include:
- Incremental (online) learning: Model updates with each new data point or mini-batch without retraining on the entire dataset.
- Concept drift detection: Use a statistical test (Kolmogorov-Smirnov test preferred) to compare recent data distribution vs. a reference window.
- Adaptive learning rate: Automatically adjust learning rate according to detected drift magnitude.
- Memory management: Retain a sliding window of streaming data for reference and drift analysis (configurable window size).
- Performance history: Track rolling prediction accuracy, loss metrics, and alert if performance degrades beyond threshold.
- Catastrophic forgetting safeguards: Use partial_fit only with algorithms that support it (e.g., SGDClassifier, PassiveAggressiveClassifier, etc. from scikit-learn), snapshot model weights periodically.
- Full model retrain: Trigger entire model retrain if drift or degradation persists for N windows.
- Monitoring utilities: Expose methods to monitor prediction accuracy over specified historical time windows.
"""

import numpy as np
import logging
from collections import deque
from scipy.stats import ks_2samp
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib

logger = logging.getLogger(__name__)

class OnlineLearner:
    def __init__(
        self,
        model=None,
        window_size=100,
        drift_threshold=0.2,
        retrain_patience=3,
        min_accuracy=0.7,
        model_save_path="online_model.pkl"
    ):
        # If no model provided, use SGDClassifier with default params
        self.model = model or SGDClassifier(loss='log_loss')
        self.window_size = window_size
        self.drift_threshold = drift_threshold  # p-value cutoff for KS test
        self.retrain_patience = retrain_patience
        self.min_accuracy = min_accuracy
        self.model_save_path = model_save_path

        self.X_reference = deque(maxlen=window_size)
        self.y_reference = deque(maxlen=window_size)
        self.accuracy_history = deque(maxlen=10)
        self.drift_history = deque(maxlen=5)
        self.drift_counter = 0

    def partial_fit(self, X_new, y_new, classes=None):
        # Supports only models with partial_fit method (e.g., SGDClassifier)
        self.model.partial_fit(X_new, y_new, classes=classes)
        # Update memory buffer
        self._update_reference(X_new, y_new)

    def _update_reference(self, X_new, y_new):
        for x, y in zip(X_new, y_new):
            self.X_reference.append(x)
            self.y_reference.append(y)

    def detect_drift(self, X_recent):
        # Use KS test: compare latest window to reference distribution
        if len(self.X_reference) < self.window_size:
            return False, 1.0  # Not enough data
        ref = np.array(self.X_reference)
        recent = np.array(X_recent)
        # Flatten if only one feature
        if ref.ndim == 1:
            statistic, p_val = ks_2samp(ref, recent)
        else:
            # Average KS test over all features
            p_vals = []
            for i in range(ref.shape[1]):
                s, p = ks_2samp(ref[:, i], recent[:, i])
                p_vals.append(p)
            p_val = np.mean(p_vals)
        drift_detected = p_val < self.drift_threshold
        self.drift_history.append(drift_detected)
        return drift_detected, p_val

    def adapt_learning_rate(self, drift_magnitude):
        """Simple adaptation: increase learning rate if high drift"""
        # SGDClassifier uses 'learning_rate' parameter, not 'eta0'
        try:
            if hasattr(self.model, 'set_params'):
                current_lr = getattr(self.model, '_learning_rate_init', 0.1)
                if drift_magnitude < 0.05:  # very low p-value: high drift
                    new_lr = min(1.0, current_lr * 1.5)
                elif drift_magnitude < 0.1:
                    new_lr = min(1.0, current_lr * 1.2)
                else:
                    new_lr = max(0.01, current_lr * 0.9)
                
                # Update the model's learning rate parameter
                self.model.set_params(learning_rate='constant', eta0=new_lr)
        except Exception as e:
            logger.warning(f"Could not adapt learning rate: {e}")
