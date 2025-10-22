from setuptools import setup, find_packages

setup(
    name='ter-ilan',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.1",
        #"torchsummary>=1.5.1",
        'matplotlib',
        'seaborn',
        "numpy",
        "pandas",
        #"scikit-learn",
        "tensorboard>=2.18.0",
        #"transformers>=4.48.1",
        # "timm",
        #"causal-conv1d>=1.1.0",
        "mamba-ssm",
        "tqdm"
    ],
    python_requires='>=3.10'
)
