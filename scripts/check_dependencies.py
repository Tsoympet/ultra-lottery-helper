#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependency Vulnerability Scanner

Scans project dependencies for known security vulnerabilities using safety.
Run this script regularly to ensure dependencies are secure.
"""

import subprocess
import sys
from pathlib import Path


def check_safety_installed():
    """Check if safety is installed."""
    try:
        subprocess.run(['safety', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_safety():
    """Install safety package."""
    print("Installing safety...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'safety'], check=True)
        print("βœ… Safety installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install safety: {e}")
        return False


def scan_dependencies(requirements_file='requirements.txt'):
    """
    Scan dependencies for vulnerabilities.
    
    Args:
        requirements_file: Path to requirements file
    """
    req_path = Path(requirements_file)
    
    if not req_path.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return 1
    
    print(f"Scanning dependencies in {requirements_file}...")
    print("=" * 60)
    
    try:
        # Run safety check
        result = subprocess.run(
            ['safety', 'check', '--file', str(req_path), '--json'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("βœ… No known security vulnerabilities found!")
            return 0
        else:
            print("⚠️  Security vulnerabilities detected:")
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            return 1
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running safety: {e}")
        return 1


def main():
    """Main entry point."""
    print("Oracle Lottery Predictor - Dependency Vulnerability Scanner")
    print("=" * 60)
    
    # Check if safety is installed
    if not check_safety_installed():
        print("Safety not found. Installing...")
        if not install_safety():
            print("\n❌ Please install safety manually: pip install safety")
            return 1
    
    # Scan main dependencies
    exit_code = scan_dependencies('requirements.txt')
    
    # Also scan dev dependencies if they exist
    if Path('requirements-dev.txt').exists():
        print("\nScanning development dependencies...")
        dev_exit_code = scan_dependencies('requirements-dev.txt')
        exit_code = max(exit_code, dev_exit_code)
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("βœ… All dependencies are secure!")
    else:
        print("⚠️  Please review and update vulnerable dependencies")
        print("Run 'pip install --upgrade <package>' to update")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
