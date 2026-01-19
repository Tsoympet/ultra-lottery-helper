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
from pathlib import Path
from typing import Any, Dict, Optional, Union


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
