#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Oracle Lottery Predictor.

Provides common utilities for:
- JSON file I/O with proper error handling
- Logging configuration
- Path operations
- Validation helpers
"""

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union


# =============================================================================
# Logging Configuration
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__ of calling module)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# =============================================================================
# File I/O Utilities
# =============================================================================

def load_json(
    path: Union[str, Path],
    default: Optional[Any] = ...,  # Use Ellipsis as sentinel
    logger: Optional[logging.Logger] = None
) -> Any:
    """
    Load JSON from file with proper error handling.
    
    Args:
        path: File path to load from
        default: Default value to return if file doesn't exist or is invalid
                 If not provided, raises ValueError on errors
        logger: Optional logger for error reporting
        
    Returns:
        Parsed JSON data or default value on error
        
    Raises:
        ValueError: If JSON is invalid and no default is provided
    """
    path = Path(path)
    
    try:
        if not path.exists():
            if logger:
                logger.debug(f"File not found: {path}, using default")
            if default is not ...:
                return default
            raise FileNotFoundError(f"File not found: {path}")
            
        content = path.read_text(encoding='utf-8')
        return json.loads(content)
        
    except json.JSONDecodeError as e:
        if logger:
            logger.error(f"Invalid JSON in {path}: {e}")
        if default is not ...:
            return default
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
        
    except Exception as e:
        if logger:
            logger.error(f"Error reading {path}: {e}")
        if default is not ...:
            return default
        raise


def save_json(
    path: Union[str, Path],
    data: Any,
    atomic: bool = True,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Save data to JSON file with proper error handling.
    
    Args:
        path: File path to save to
        data: Data to serialize to JSON
        atomic: If True, write to temp file first then rename (prevents corruption)
        logger: Optional logger for error reporting
        
    Raises:
        ValueError: If data cannot be serialized to JSON
        IOError: If file cannot be written
    """
    path = Path(path)
    
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize to JSON
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        if atomic:
            # Write to temp file first, then rename
            temp_path = path.with_suffix(path.suffix + '.tmp')
            temp_path.write_text(json_str, encoding='utf-8')
            temp_path.replace(path)
        else:
            path.write_text(json_str, encoding='utf-8')
            
        if logger:
            logger.debug(f"Saved JSON to {path}")
            
    except (TypeError, ValueError) as e:
        if logger:
            logger.error(f"Cannot serialize data to JSON: {e}")
        raise ValueError(f"Cannot serialize data to JSON: {e}") from e
        
    except Exception as e:
        if logger:
            logger.error(f"Error writing {path}: {e}")
        raise IOError(f"Error writing {path}: {e}") from e


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    name: str = "value"
) -> None:
    """
    Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the value for error messages
        
    Raises:
        ValueError: If value is out of range
    """
    if not min_val <= value <= max_val:
        raise ValueError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )


def validate_positive(value: Union[int, float], name: str = "value") -> None:
    """
    Validate that a value is positive (> 0).
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def ensure_path_exists(path: Union[str, Path], is_file: bool = False) -> Path:
    """
    Ensure a path exists, creating directories if needed.
    
    Args:
        path: Path to ensure exists
        is_file: If True, ensure parent directory exists; if False, ensure path itself exists
        
    Returns:
        Path object
    """
    path = Path(path)
    
    if is_file:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
        
    return path


# =============================================================================
# Network Utilities
# =============================================================================

import time
from collections import deque
from typing import Callable, TypeVar

T = TypeVar('T')


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.
    
    Limits the number of operations within a time window.
    """
    
    def __init__(self, max_requests: int, window_seconds: float):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove old requests outside the window
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()
        
        # If at limit, wait until oldest request expires
        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] + self.window_seconds - now
            if sleep_time > 0:
                time.sleep(sleep_time)
                # Clean up after waiting
                while self.requests and self.requests[0] < time.time() - self.window_seconds:
                    self.requests.popleft()
        
        # Record this request
        self.requests.append(time.time())


def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable[..., T]:
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_multiplier: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
        logger: Optional logger for retry notifications
        
    Returns:
        Wrapped function with retry logic
    """
    def wrapper(*args, **kwargs) -> T:
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                
                if attempt < max_retries:
                    if logger:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                    time.sleep(delay)
                    delay *= backoff_multiplier
                else:
                    if logger:
                        logger.error(f"All {max_retries + 1} attempts failed")
        
        # If we get here, all retries failed
        raise last_exception
    
    return wrapper


# =============================================================================
# Data Validation with JSON Schema
# =============================================================================

# Try to import jsonschema, but make it optional
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


def validate_json_schema(
    data: Any,
    schema: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate data against a JSON Schema.
    
    Args:
        data: Data to validate
        schema: JSON Schema definition
        logger: Optional logger for validation errors
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ImportError: If jsonschema library is not installed
    """
    if not JSONSCHEMA_AVAILABLE:
        raise ImportError(
            "jsonschema library not installed. "
            "Install with: pip install jsonschema"
        )
    
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        if logger:
            logger.error(f"JSON schema validation failed: {e.message}")
            if e.path:
                logger.error(f"Path: {'.'.join(str(p) for p in e.path)}")
        return False
    except jsonschema.SchemaError as e:
        if logger:
            logger.error(f"Invalid schema: {e.message}")
        return False


# Common JSON schemas for lottery data
LOTTERY_DRAW_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string", "format": "date"},
        "n1": {"type": "integer", "minimum": 1},
        "n2": {"type": "integer", "minimum": 1},
        "n3": {"type": "integer", "minimum": 1},
        "n4": {"type": "integer", "minimum": 1},
        "n5": {"type": "integer", "minimum": 1},
        "n6": {"type": "integer", "minimum": 1},
        "e1": {"type": "integer", "minimum": 1},
        "e2": {"type": "integer", "minimum": 1},
        "joker": {"type": "integer", "minimum": 1}
    },
    "required": ["date"]
}

PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "game": {"type": "string", "minLength": 1},
        "draw_date": {"type": "string", "format": "date"},
        "numbers": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1}
        },
        "secondary_numbers": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1}
        }
    },
    "required": ["game", "numbers"]
}
