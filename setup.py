from setuptools import setup, find_packages
import subprocess
import sys

def check_near_cli():
    try:
        subprocess.run(['near', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nWARNING: NEAR CLI not found!")
        print("Please install it with: npm install -g near-cli\n")

if not sys.argv[1] == "egg_info":  # Don't check during pip operations
    check_near_cli()

setup(
    name="agent-arcade",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Click>=8.0",
        "stable-baselines3>=2.0.0",
        "gymnasium[atari]>=0.29.0",
        "near-api-py>=0.3.0",
        "PyYAML>=6.0.1",
        "wandb>=0.15.0",
        "opencv-python>=4.8.0",
    ],
    entry_points={
        "console_scripts": [
            "agent-arcade=cli.main:main",
        ],
    },
    python_requires=">=3.8",
    author="Agent Arcade Team",
    description="Train and compete with AI agents on the NEAR blockchain",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jbarnes850/agent-arcade",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
) 