from setuptools import setup, find_packages

setup(
    name="agent-arcade",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Click>=8.0",
        "stable-baselines3==2.4.1",
        "gymnasium[atari]==0.29.1",
        "ale-py==0.8.1",
        "AutoROM[accept-rom-license]==0.6.1",
        "PyYAML>=6.0.1",
        "wandb>=0.15.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "tensorboard>=2.13.0"
    ],
    entry_points={
        "console_scripts": [
            "agent-arcade=cli.main:cli",
        ],
    },
    python_requires=">=3.8",
    author="Agent Arcade Team",
    description="Train and compete with AI agents using deep reinforcement learning",
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