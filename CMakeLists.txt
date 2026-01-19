# CMakeLists.txt for Oracle Lottery Predictor
# This file provides CMake build support for the Python project
# Useful for C/C++ integration and cross-platform builds

cmake_minimum_required(VERSION 3.15)
project(OracleLotteryPredictor VERSION 6.3.0 LANGUAGES NONE)

# Find Python
find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)

# Set project variables
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(PROJECT_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
set(PROJECT_ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)

# Installation prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Installation directory" FORCE)
endif()

# Option to install in development mode
option(INSTALL_DEV_MODE "Install in development/editable mode" ON)

# Custom target for installing Python dependencies
add_custom_target(install-deps
    COMMAND ${Python3_EXECUTABLE} -m pip install --upgrade pip
    COMMAND ${Python3_EXECUTABLE} -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Installing Python dependencies..."
)

# Custom target for installing development dependencies
add_custom_target(install-dev-deps
    COMMAND ${Python3_EXECUTABLE} -m pip install --upgrade pip
    COMMAND ${Python3_EXECUTABLE} -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
    COMMAND ${Python3_EXECUTABLE} -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements-dev.txt
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Installing development dependencies..."
)

# Custom target for installing the package
add_custom_target(install-package
    COMMAND ${Python3_EXECUTABLE} -m pip install -e .
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Installing Oracle Lottery Predictor package..."
    DEPENDS install-deps
)

# Custom target for running tests
add_custom_target(test
    COMMAND ${CMAKE_COMMAND} -E env QT_QPA_PLATFORM=offscreen ${Python3_EXECUTABLE} -m pytest -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running tests..."
)

# Custom target for linting
add_custom_target(lint
    COMMAND ${Python3_EXECUTABLE} -m flake8 src/ --max-line-length=100 --ignore=E203,W503 || true
    COMMAND ${Python3_EXECUTABLE} -m mypy src/ || true
    COMMAND ${Python3_EXECUTABLE} -m bandit -r src/ || true
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running linters..."
)

# Custom target for code formatting
add_custom_target(format
    COMMAND ${Python3_EXECUTABLE} -m black src/ tests/
    COMMAND ${Python3_EXECUTABLE} -m isort src/ tests/
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Formatting code..."
)

# Custom target for cleaning build artifacts
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E rm -rf build dist dist_installer
    COMMAND ${CMAKE_COMMAND} -E rm -rf htmlcov .coverage .pytest_cache .mypy_cache
    COMMAND ${Python3_EXECUTABLE} -c "import os, shutil; [shutil.rmtree(os.path.join(root, d)) for root, dirs, _ in os.walk('.') for d in dirs if d == '__pycache__']"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Cleaning build artifacts..."
)

# Custom target for running the desktop application
add_custom_target(run
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/ulh_desktop.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running Oracle Lottery Predictor desktop application..."
)

# Install Python source files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/src
    FILES_MATCHING PATTERN "*.py"
)

# Install data files
install(DIRECTORY ${PROJECT_DATA_DIR}/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/data
    OPTIONAL
)

# Install assets
install(DIRECTORY ${PROJECT_ASSETS_DIR}/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/assets
    OPTIONAL
)

# Install configuration files
install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)

# Print configuration summary
message(STATUS "Oracle Lottery Predictor ${PROJECT_VERSION}")
message(STATUS "Python: ${Python3_EXECUTABLE} (${Python3_VERSION})")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Source directory: ${PROJECT_SOURCE_DIR}")
message(STATUS "")
message(STATUS "Available targets:")
message(STATUS "  install-deps      - Install Python dependencies")
message(STATUS "  install-dev-deps  - Install development dependencies")
message(STATUS "  install-package   - Install the package")
message(STATUS "  test              - Run test suite")
message(STATUS "  lint              - Run linters")
message(STATUS "  format            - Format code")
message(STATUS "  clean-all         - Clean build artifacts")
message(STATUS "  run               - Run the application")
