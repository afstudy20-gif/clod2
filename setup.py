from setuptools import find_packages, setup

setup(
    name="cclaude",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.40.0",
        "openai>=1.50.0",
        "google-generativeai>=0.8.0",
        "rich>=13.0.0",
        "click>=8.1.0",
        "prompt_toolkit>=3.0.0",
        "pygments>=2.18.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "cclaude=main:main",
        ],
    },
    python_requires=">=3.11",
)
