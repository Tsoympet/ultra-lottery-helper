# Docker and CMake Usage Guide

This guide covers using Docker and CMake to build, run, and deploy the Oracle Lottery Predictor application.

## Table of Contents

- [Docker Usage](#docker-usage)
  - [Building the Docker Image](#building-the-docker-image)
  - [Running with Docker](#running-with-docker)
  - [Docker Compose](#docker-compose)
  - [GUI Support on Linux](#gui-support-on-linux)
- [CMake Build System](#cmake-build-system)
  - [Prerequisites](#prerequisites)
  - [Building with CMake](#building-with-cmake)
  - [Available CMake Targets](#available-cmake-targets)
  - [CMake Installation](#cmake-installation)
- [Troubleshooting](#troubleshooting)

---

## Docker Usage

### Building the Docker Image

Build the Docker image from the project root:

```bash
docker build -t oracle-lottery-predictor .
```

The Dockerfile uses a multi-stage build to optimize image size and includes all necessary dependencies for running the application.

### Running with Docker

#### Run the desktop application (headless mode)

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/exports:/app/exports \
  oracle-lottery-predictor python src/ulh_desktop.py
```

#### Run the learning CLI

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/exports:/app/exports \
  oracle-lottery-predictor python src/ulh_learn_cli.py --help
```

#### Run the data fetcher

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  oracle-lottery-predictor python src/lottery_data_fetcher.py --help
```

### Docker Compose

Docker Compose provides a simpler way to manage containers with predefined configurations.

#### Start the main application

```bash
docker-compose up oracle-lottery
```

#### Run learning CLI (with tools profile)

```bash
docker-compose --profile tools run oracle-lottery-learn
```

#### Run data fetcher (with tools profile)

```bash
docker-compose --profile tools run oracle-lottery-fetcher
```

#### Build all services

```bash
docker-compose build
```

#### Stop all services

```bash
docker-compose down
```

### GUI Support on Linux

To run the desktop UI with display support on Linux:

1. Allow X11 connections:
```bash
xhost +local:docker
```

2. Run with X11 forwarding:
```bash
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/exports:/app/exports \
  --network host \
  oracle-lottery-predictor python src/ulh_desktop.py
```

3. When done, revoke X11 access:
```bash
xhost -local:docker
```

#### Alternative: Using docker-compose for GUI

Uncomment the GUI-related environment variables and volumes in `docker-compose.yml`:

```yaml
environment:
  - DISPLAY=${DISPLAY}
  - QT_X11_NO_MITSHM=1
volumes:
  - /tmp/.X11-unix:/tmp/.X11-unix:rw
network_mode: host
```

Then run:
```bash
xhost +local:docker
docker-compose up oracle-lottery
xhost -local:docker
```

---

## CMake Build System

CMake provides a cross-platform build system for the project, making it easy to build, test, and install on different platforms.

### Prerequisites

- CMake 3.15 or higher
- Python 3.9 or higher

Install CMake:

```bash
# Ubuntu/Debian
sudo apt-get install cmake

# macOS
brew install cmake

# Windows
# Download from https://cmake.org/download/
```

### Building with CMake

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Configure the project:
```bash
cmake ..
```

3. Build specific targets (see Available CMake Targets below):
```bash
cmake --build . --target install-deps
cmake --build . --target install-package
```

### Available CMake Targets

#### `install-deps`
Install Python dependencies from `requirements.txt`:
```bash
cmake --build . --target install-deps
```

#### `install-dev-deps`
Install both runtime and development dependencies:
```bash
cmake --build . --target install-dev-deps
```

#### `install-package`
Install the Oracle Lottery Predictor package in editable mode:
```bash
cmake --build . --target install-package
```

#### `test`
Run the test suite with pytest:
```bash
cmake --build . --target test
```

#### `lint`
Run code linters (flake8, mypy, bandit):
```bash
cmake --build . --target lint
```

#### `format`
Format code with black and isort:
```bash
cmake --build . --target format
```

#### `clean-all`
Clean all build artifacts and caches:
```bash
cmake --build . --target clean-all
```

#### `run`
Run the desktop application:
```bash
cmake --build . --target run
```

### CMake Installation

By default, CMake will install files to `./install` directory. You can customize the installation prefix:

```bash
# Configure with custom install prefix
cmake -DCMAKE_INSTALL_PREFIX=/opt/oracle-lottery ..

# Install files
cmake --build . --target install
```

This will install:
- Python source files to `${CMAKE_INSTALL_PREFIX}/src`
- Data files to `${CMAKE_INSTALL_PREFIX}/data`
- Assets to `${CMAKE_INSTALL_PREFIX}/assets`
- Configuration files to `${CMAKE_INSTALL_PREFIX}`

### Complete CMake Workflow

Here's a complete workflow from setup to running:

```bash
# 1. Create build directory
mkdir build && cd build

# 2. Configure project
cmake ..

# 3. Install dependencies
cmake --build . --target install-deps

# 4. Install package
cmake --build . --target install-package

# 5. Run tests (optional)
cmake --build . --target test

# 6. Run the application
cmake --build . --target run
```

---

## Troubleshooting

### Docker Issues

#### SSL Certificate Errors During Build

If you encounter SSL certificate errors during `pip install` in the Docker build, you may need to:

1. Check your network/firewall settings
2. Use a different base image
3. Add trusted certificates to the Docker build

#### Permission Errors with Mounted Volumes

If you get permission errors with mounted volumes:

```bash
# On Linux, ensure proper ownership
sudo chown -R $USER:$USER data/ exports/

# Or run with user mapping
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/exports:/app/exports \
  oracle-lottery-predictor python src/ulh_desktop.py
```

#### Container Immediately Exits

The default CMD shows help. To keep the container running or execute specific commands:

```bash
# Override the default command
docker run --rm oracle-lottery-predictor python src/ulh_desktop.py

# Or use interactive mode
docker run -it oracle-lottery-predictor /bin/bash
```

### CMake Issues

#### CMake Not Found

Install CMake using your package manager (see Prerequisites section above).

#### Python Not Found

Ensure Python 3.9+ is installed and in your PATH:

```bash
python3 --version
```

#### Build Directory Conflicts

If you encounter build issues, clean the build directory:

```bash
rm -rf build/
mkdir build
cd build
cmake ..
```

#### Module Import Errors After CMake Build

Ensure you've run the `install-package` target:

```bash
cd build
cmake --build . --target install-package
```

### General Issues

#### Qt Platform Plugin Errors

On headless systems, ensure the `QT_QPA_PLATFORM` environment variable is set:

```bash
export QT_QPA_PLATFORM=offscreen
```

This is already set in the Dockerfile, but may be needed for CMake builds.

#### Data or Exports Directory Not Found

Create the required directories:

```bash
mkdir -p data/history exports
```

---

## Additional Resources

- [Main README](README.md) - Project overview and features
- [SETUP.md](SETUP.md) - Detailed setup instructions
- [DATA_FETCHER_README.md](DATA_FETCHER_README.md) - Data fetching documentation
- [SCHEDULER_README.md](SCHEDULER_README.md) - Scheduling documentation
- [PREDICTION_TRACKER_README.md](PREDICTION_TRACKER_README.md) - Prediction tracking documentation
- [AI_SYSTEM_STATUS.md](AI_SYSTEM_STATUS.md) - AI/ML system status

## Contributing

When contributing changes that affect Docker or CMake builds:

1. Test the Docker build locally
2. Test relevant CMake targets
3. Update this documentation if needed
4. Ensure `.dockerignore` excludes unnecessary files

## License

MIT License - see [LICENSE.txt](LICENSE.txt) for details.
