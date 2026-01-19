# Advanced Security Features

This document describes the advanced security features implemented in Oracle Lottery Predictor.

## Overview

The following security enhancements have been added to the codebase:

1. **Rate Limiting** - Prevents overwhelming external services
2. **Request Retry with Exponential Backoff** - Handles transient failures gracefully
3. **JSON Schema Validation** - Ensures data integrity
4. **Security Headers** - Ready for future web interfaces
5. **Automated Dependency Scanning** - Regular vulnerability checks

---

## 1. Rate Limiting

### Purpose
Prevents the application from making too many requests to external services in a short time period, avoiding rate limit violations and ensuring fair usage.

### Implementation
Uses a token bucket algorithm to limit requests per time window.

### Configuration
Located in `src/config.py`:

```python
class NetworkConfig:
    RATE_LIMIT_ENABLED = True  # Enable/disable rate limiting
    RATE_LIMIT_REQUESTS = 10   # Max requests per window
    RATE_LIMIT_WINDOW = 60     # Window size in seconds
```

### Usage Example

```python
from src.utils import RateLimiter

# Create rate limiter (10 requests per 60 seconds)
rate_limiter = RateLimiter(
    max_requests=10,
    window_seconds=60
)

# Before making a request
rate_limiter.wait_if_needed()
response = requests.get(url)
```

### How It Works
- Tracks timestamps of recent requests
- Automatically waits if rate limit would be exceeded
- Old requests outside the window are discarded
- Thread-safe for concurrent usage

---

## 2. Request Retry with Exponential Backoff

### Purpose
Handles transient network failures by automatically retrying failed requests with increasing delays, improving reliability.

### Configuration
Located in `src/config.py`:

```python
class NetworkConfig:
    MAX_RETRIES = 3              # Maximum retry attempts
    INITIAL_RETRY_DELAY = 1.0    # Initial delay in seconds
    RETRY_BACKOFF = 2.0          # Multiplier for exponential backoff
```

### Usage Example

```python
from src.utils import retry_with_backoff
import requests

def fetch_data(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

# Wrap function with retry logic
fetch_with_retry = retry_with_backoff(
    fetch_data,
    max_retries=3,
    initial_delay=1.0,
    backoff_multiplier=2.0,
    exceptions=(requests.exceptions.RequestException,),
    logger=logger
)

# Use it
data = fetch_with_retry("https://api.example.com/data")
```

### Retry Sequence
For default configuration (3 retries, 1s initial, 2x backoff):
1. **First attempt**: Immediate
2. **First retry**: Wait 1 second
3. **Second retry**: Wait 2 seconds
4. **Third retry**: Wait 4 seconds
5. **Final attempt**: Raise exception if all fail

### Integration
The `fetch_online_history()` function in `ultra_lottery_helper.py` now uses retry logic automatically:

```python
def fetch_online_history(game: str):
    # Automatically retries with exponential backoff
    # Applies rate limiting
    # Logs retry attempts
    ...
```

---

## 3. JSON Schema Validation

### Purpose
Validates JSON data structures against predefined schemas to ensure data integrity and catch format errors early.

### Installation
```bash
pip install jsonschema
```

Or add to your environment:
```bash
pip install -r requirements-dev.txt
```

### Predefined Schemas

#### Lottery Draw Schema
```python
from src.utils import LOTTERY_DRAW_SCHEMA, validate_json_schema

draw_data = {
    "date": "2026-01-19",
    "n1": 5,
    "n2": 12,
    "n3": 18,
    "n4": 27,
    "n5": 33
}

is_valid = validate_json_schema(draw_data, LOTTERY_DRAW_SCHEMA, logger=logger)
```

#### Prediction Schema
```python
from src.utils import PREDICTION_SCHEMA, validate_json_schema

prediction = {
    "game": "EUROJACKPOT",
    "numbers": [5, 12, 18, 27, 33],
    "secondary_numbers": [2, 8]
}

is_valid = validate_json_schema(prediction, PREDICTION_SCHEMA, logger=logger)
```

### Custom Schemas

```python
from src.utils import validate_json_schema

# Define custom schema
config_schema = {
    "type": "object",
    "properties": {
        "timeout": {"type": "integer", "minimum": 1, "maximum": 60},
        "retries": {"type": "integer", "minimum": 0, "maximum": 10}
    },
    "required": ["timeout"]
}

# Validate data
config = {"timeout": 10, "retries": 3}
if validate_json_schema(config, config_schema, logger=logger):
    print("Configuration is valid!")
```

### Error Handling
- Returns `False` on validation failure
- Logs detailed error messages if logger provided
- Shows the exact path where validation failed

---

## 4. Security Headers

### Purpose
Provides security headers configuration for future web interfaces (e.g., if adding a web dashboard).

### Configuration
Located in `src/config.py`:

```python
class NetworkConfig:
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }
```

### Usage (Future Web Interface)

```python
from src.config import NetworkConfig
from flask import Flask, make_response

app = Flask(__name__)

@app.after_request
def add_security_headers(response):
    for header, value in NetworkConfig.SECURITY_HEADERS.items():
        response.headers[header] = value
    return response
```

### Headers Explained

| Header | Value | Purpose |
|--------|-------|---------|
| `X-Content-Type-Options` | `nosniff` | Prevents MIME type sniffing |
| `X-Frame-Options` | `DENY` | Prevents clickjacking attacks |
| `X-XSS-Protection` | `1; mode=block` | Enables XSS filter |
| `Strict-Transport-Security` | `max-age=31536000` | Forces HTTPS for 1 year |

---

## 5. Automated Dependency Scanning

### Purpose
Regularly scans project dependencies for known security vulnerabilities using the Safety database.

### Manual Scanning

#### Using the Script
```bash
# Run the dependency scanner
python scripts/check_dependencies.py
```

The script will:
1. Check if `safety` is installed
2. Install it if needed
3. Scan `requirements.txt`
4. Scan `requirements-dev.txt` (if exists)
5. Report any vulnerabilities found

#### Using Safety Directly
```bash
# Install safety
pip install safety

# Scan dependencies
safety check --file requirements.txt

# Generate JSON report
safety check --file requirements.txt --json > vulnerabilities.json
```

### Automated Scanning

#### GitHub Actions Workflow
A workflow is configured in `.github/workflows/dependency-scan.yml` that:

- **Runs weekly**: Every Monday at 9:00 AM UTC
- **Runs on changes**: When requirements files are modified
- **Can be triggered manually**: Via GitHub Actions UI

#### Viewing Results
1. Go to the **Actions** tab in your GitHub repository
2. Select **Security - Dependency Scan** workflow
3. View the latest run results
4. Check the summary for vulnerabilities

### Responding to Vulnerabilities

When vulnerabilities are found:

1. **Review the report** - Understand the severity and impact
2. **Check for updates** - See if a patched version is available
3. **Update the dependency**:
   ```bash
   pip install --upgrade vulnerable-package
   pip freeze > requirements.txt
   ```
4. **Test the application** - Ensure the update doesn't break functionality
5. **Commit the update**:
   ```bash
   git add requirements.txt
   git commit -m "Security: Update vulnerable-package to vX.Y.Z"
   ```

### Exclusions
If a vulnerability cannot be fixed immediately (e.g., no patch available):

```bash
# Create a safety policy file
safety check --file requirements.txt --save-policy safety-policy.yml

# Edit safety-policy.yml to ignore specific CVEs (with justification)
```

---

## Configuration Summary

All security features can be configured in `src/config.py`:

```python
class NetworkConfig:
    # Request configuration
    REQUEST_TIMEOUT = 10
    VERIFY_SSL = True
    
    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1.0
    RETRY_BACKOFF = 2.0
    
    # Rate limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_REQUESTS = 10
    RATE_LIMIT_WINDOW = 60
    
    # Security headers
    SECURITY_HEADERS = { ... }
```

---

## Best Practices

1. **Rate Limiting**
   - Keep `RATE_LIMIT_ENABLED = True` in production
   - Adjust limits based on API provider requirements
   - Monitor logs for rate limit warnings

2. **Retry Logic**
   - Don't set `MAX_RETRIES` too high (avoid long waits)
   - Use appropriate backoff multipliers (2.0 is standard)
   - Log retry attempts for debugging

3. **Schema Validation**
   - Validate all external data before processing
   - Define schemas for all critical data structures
   - Update schemas when data format changes

4. **Dependency Scanning**
   - Run scans weekly (automated via GitHub Actions)
   - Review and address vulnerabilities promptly
   - Keep dependencies up to date

5. **Security Headers**
   - Apply to all web interfaces
   - Test headers with security scanning tools
   - Keep updated with latest security recommendations

---

## Testing

### Test Rate Limiting
```python
from src.utils import RateLimiter
import time

limiter = RateLimiter(max_requests=3, window_seconds=5)

start = time.time()
for i in range(5):
    limiter.wait_if_needed()
    print(f"Request {i+1} at {time.time() - start:.2f}s")
# Should wait after 3rd request
```

### Test Retry Logic
```python
from src.utils import retry_with_backoff

attempt_count = 0

def flaky_function():
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ValueError("Simulated failure")
    return "Success"

retry_func = retry_with_backoff(
    flaky_function,
    max_retries=3,
    initial_delay=0.1
)

result = retry_func()  # Should succeed on 3rd attempt
```

### Test Schema Validation
```python
from src.utils import validate_json_schema, LOTTERY_DRAW_SCHEMA

# Valid data
valid_draw = {"date": "2026-01-19", "n1": 5, "n2": 12, "n3": 18, "n4": 27, "n5": 33}
assert validate_json_schema(valid_draw, LOTTERY_DRAW_SCHEMA)

# Invalid data (missing date)
invalid_draw = {"n1": 5, "n2": 12}
assert not validate_json_schema(invalid_draw, LOTTERY_DRAW_SCHEMA)
```

---

## Troubleshooting

### Issue: Rate limiter blocking too aggressively
**Solution**: Increase `RATE_LIMIT_REQUESTS` or `RATE_LIMIT_WINDOW` in config

### Issue: Retries taking too long
**Solution**: Reduce `MAX_RETRIES` or decrease `RETRY_BACKOFF` multiplier

### Issue: jsonschema not found
**Solution**: Install it: `pip install jsonschema`

### Issue: Safety check fails
**Solution**: Update safety: `pip install --upgrade safety`

---

## Security Checklist

- [x] Rate limiting enabled for all external requests
- [x] Retry logic with exponential backoff implemented
- [x] JSON schema validation available for data validation
- [x] Security headers configured for future web interfaces
- [x] Automated dependency scanning via GitHub Actions
- [x] Manual dependency scanning script available
- [x] All configurations centralized in `config.py`
- [x] Comprehensive documentation provided

---

For questions or issues, please open a GitHub issue with the "security" label.
