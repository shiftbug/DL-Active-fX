#!/usr/bin/env python3
# run.py
# Launcher script for the Deep Learning Activation Function Comparison app

import os
import sys
import subprocess
import logging
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 7)
    current_version = sys.version_info
    
    if current_version < required_version:
        logger.error(f"Python {required_version[0]}.{required_version[1]} or higher is required.")
        logger.error(f"Current version: {current_version[0]}.{current_version[1]}")
        sys.exit(1)
    
    logger.info(f"Python version: {sys.version.split()[0]}")

def check_requirements():
    """Check if necessary packages are installed and install if missing"""
    required_packages = [
        "streamlit>=1.20.0",
        "tensorflow>=2.9.0",
        "numpy>=1.19.5",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0"
    ]
    
    # Check for Apple Silicon and add TF Metal if needed
    if platform.system() == "Darwin" and platform.machine().startswith("arm"):
        logger.info("Apple Silicon detected, adding Metal support packages")
        required_packages.extend(["tensorflow-macos>=2.9.0", "tensorflow-metal>=0.5.0"])
    
    # Try to import each package
    missing_packages = []
    for package in required_packages:
        package_name = package.split(">=")[0]
        try:
            __import__(package_name)
            logger.info(f"Package {package_name} is already installed")
        except ImportError:
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            logger.info("Package installation complete!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e}")
            sys.exit(1)
    else:
        logger.info("All required packages are already installed")

def check_modules_directory():
    """Check if modules directory exists and create if necessary"""
    modules_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
    
    if not os.path.exists(modules_dir):
        logger.warning(f"Modules directory not found at {modules_dir}")
        try:
            os.makedirs(modules_dir)
            # Create empty __init__.py file
            with open(os.path.join(modules_dir, "__init__.py"), "w") as f:
                pass
            logger.info("Created modules directory and __init__.py file")
        except OSError as e:
            logger.error(f"Failed to create modules directory: {e}")
            sys.exit(1)

def run_app():
    """Run the Streamlit app"""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    # Make sure the app file exists
    if not os.path.exists(app_path):
        logger.error(f"Main application file not found at {app_path}")
        sys.exit(1)
    
    # Run the Streamlit app
    logger.info("Starting Streamlit app...")
    try:
        subprocess.call(["streamlit", "run", app_path])
    except Exception as e:
        logger.error(f"Failed to start Streamlit app: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    logger.info("Setting up Deep Learning Activation Function Comparison app...")
    check_python_version()
    check_requirements()
    check_modules_directory()
    run_app()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)