#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration constants for Oracle Lottery Predictor.

Centralizes all magic numbers and configuration values used throughout the application.
"""

import math


# =============================================================================
# Algorithm Configuration
# =============================================================================

class AlgorithmConfig:
    """Algorithm-specific constants and thresholds."""
    
    # EWMA (Exponentially Weighted Moving Average)
    EWMA_LOG_BASE = math.log(2.0)  # Half-life calculation denominator
    MIN_HALF_LIFE = 1  # Minimum half-life value
    
    # K-Means Clustering
    KMEANS_DEFAULT_CLUSTERS = 3  # Default number of clusters for pattern detection
    
    # Model Scoring
    MODEL_SCORE_THRESHOLD = 0.1  # Minimum score threshold for model validity
    MIN_CV_SCORE = 0.1  # Minimum cross-validation score
    
    # Model Requirements
    MIN_ML_HISTORY = 120  # Minimum data points required for ML models
    MIN_PROPHET_HISTORY = 120  # Minimum data points required for Prophet model
    
    # Probability Smoothing
    PROB_SMOOTHING_EPSILON = 1e-6  # Epsilon for probability smoothing
    PROB_SMOOTHING_SMALL = 1e-3  # Small value for probability adjustments
    
    # Baseline Multipliers
    BASELINE_WITH_SECONDARY = 1.2  # Baseline multiplier when secondary numbers exist
    BASELINE_WITHOUT_SECONDARY = 1.4  # Baseline multiplier for main numbers only
    
    # Sampling
    GUMBEL_TEMPERATURE_DEFAULT = 1.0  # Default temperature for Gumbel sampling
    MAX_SAMPLING_ATTEMPTS = 1000  # Maximum attempts for constrained sampling


# =============================================================================
# Network Configuration
# =============================================================================

class NetworkConfig:
    """Network request configuration."""
    
    REQUEST_TIMEOUT = 10  # Timeout in seconds for HTTP requests
    VERIFY_SSL = True  # Verify SSL certificates
    MAX_RETRIES = 3  # Maximum number of retry attempts
    RETRY_BACKOFF = 2.0  # Exponential backoff multiplier
    INITIAL_RETRY_DELAY = 1.0  # Initial delay in seconds before first retry
    
    # Rate limiting
    RATE_LIMIT_ENABLED = True  # Enable rate limiting
    RATE_LIMIT_REQUESTS = 10  # Maximum requests per time window
    RATE_LIMIT_WINDOW = 60  # Time window in seconds
    
    # Security headers for future web interfaces
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }


# =============================================================================
# File I/O Configuration
# =============================================================================

class FileConfig:
    """File operation configuration."""
    
    DEFAULT_ENCODING = 'utf-8'  # Default file encoding
    USE_ATOMIC_WRITES = True  # Use atomic writes for safety
    
    # Export settings
    EXPORT_CSV_COLUMNS = 6  # Number of columns in CSV export
    

# =============================================================================
# Validation Configuration
# =============================================================================

class ValidationConfig:
    """Input validation limits."""
    
    # Scheduler intervals
    MIN_INTERVAL_HOURS = 1
    MAX_INTERVAL_HOURS = 24 * 30  # 30 days
    
    # Data quality
    MIN_HISTORY_ROWS = 10  # Minimum historical data rows required
    
    # Number ranges (game-specific, these are maximums)
    MAX_LOTTERY_NUMBER = 100  # Maximum lottery number across all games
    MIN_LOTTERY_NUMBER = 1  # Minimum lottery number


# =============================================================================
# UI Configuration
# =============================================================================

class UIConfig:
    """User interface configuration."""
    
    # Window dimensions
    DEFAULT_WINDOW_WIDTH = 1000
    DEFAULT_WINDOW_HEIGHT = 800
    
    # Plot caching
    ENABLE_PLOT_CACHE = True
    
    # Update delays
    SLIDER_DEBOUNCE_MS = 300  # Debounce delay for sliders in milliseconds
