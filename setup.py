# BBT/setup.py
from setuptools import setup, find_packages

setup(
    name="bbtransformer",
    version="0.1.0",
    packages=find_packages(),  # finds 'bbtransformer/'
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "seaborn",
        "optuna>=3.0",         
        "pytorch_optimizer",
        "nilearn",              
    ],
    package_data={"bbtransformer": ["roi_labels.csv"]}, 
    include_package_data=True,
)