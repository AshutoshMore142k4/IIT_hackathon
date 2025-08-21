# Streamlit component for realtime charts
# File: /dashboard/components/realtime_charts.py

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeCharts:
    """
    Interactive Plotly-based chart components for real-time credit intelligence dashboard.
    Optimized for performance, mobile responsiveness, and accessibility.
    """
    
    def __init__(self, theme: str = 'light'):
        """
        Initialize the RealtimeCharts class.
        
        Args:
            theme: UI theme ('light' or 'dark')
        """
        self.theme = theme
        self.colors = self._get_color_palette()
        self.chart_config = self._get_chart_config()
        
        # Cache for chart data to improve performance
        self._chart_cache = {}
        
        logger.info("RealtimeCharts initialized successfully")
    
    def _get_color_palette(self) -> Dict[str, str]:
        """Get consistent color palette based on theme."""
        if self.theme == 'dark':
            return {
                'primary': '#00D4AA',
                'secondary': '#FF6B6B', 
                'success': '#4ECDC4',
                'warning': '#FFE66D',
                'danger': '#FF6B6B',
                'info': '#4A90E2',
                'background': '#1E1E1E',
                'text': '#FFFFFF',
                'grid': '#333333',
                'risk_low': '#4ECDC4',
                'risk_medium': '#FFE66D',
                'risk_high': '#FF6B6B'
            }
        else:
            return {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'success': '#059669',
                'warning': '#DC2626',
                'danger': '#B91C1C',
                'info': '#2563EB',
                'background': '#FFFFFF',
                'text': '#1F2937',
                'grid': '#E5E7EB',
                'risk_low': '#059669',
                'risk_medium': '#DC2626',
                'risk_high': '#B91C1C'
            }
    
    def _get_chart_config(self) -> Dict[str, Any]:
        """Get default chart configuration."""
        return {
            'responsive': True,
            'maintainAspectRatio': False,
            'animation': {
                'duration': 300,
                'easing': 'easeInOutQuart'
            },
            'plugins': {
                'legend': {
                    'display': True,
                    'position': 'top',
                    'labels': {
                        'color': self.colors['text'],
                        'font': {
                            'size': 12
                        }
                    }
                },
                'tooltip': {
                    'backgroundColor': self.colors['background'],
                    'titleColor': self.colors['text'],
                    'bodyColor': self.colors['text'],
                    'borderColor': self.colors['grid'],
                    'borderWidth': 1
                }
            },
            'scales': {
                'x': {
                    'grid': {
                        'color': self.colors['grid'],
                        'borderColor': self.colors['grid']
                    },
                    'ticks': {
                        'color': self.colors['text']
                    }
                },
                'y': {
                    'grid': {
                        'color': self.colors['grid'],
                        'borderColor': self.colors['grid']
                    },
                    'ticks': {
                        'color': self.colors['text']
                    }
                }
            }
        }
