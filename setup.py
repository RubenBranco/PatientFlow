from setuptools import setup
import re
from pathlib import Path


def get_version():
    with open(Path(__file__).parent / "patientflow" / "__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return re.split(r"['\"]", line)[1]


setup(
    name="patientflow",
    version=get_version(),
    description="A package containing Flow Matching models to generate mixed-type longitudinal clinical data.",
    author="Ruben Branco, Piero Fariselli, Sara Madeira",
    author_email="rmbranco@fc.ul.pt",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
    keywords="mixed-type longitudinal clinical data, generative models, deep learning",
    packages=[
        "patientflow",
        "patientflow.models",
    ],
    install_requires=[
        "torch==2.3.1",
        "lightning==2.2.2",
        "numpy==1.23.5",
        "pandas==1.5.3",
        "torchdiffeq==0.2.2",
        "scikit-learn==1.4.2",
        "torchcfm==1.0.5",
    ],
    extras_requires={
        "experiments": [
            "wandb==0.17.4",
            "rpy2==3.5.16",
            "seaborn==0.13.2",
            "msas-pytorch",
            "matplotlib==3.9.0",
            "tslearn==0.6.3",
            "numba==0.59.1"
            "ipykernel",
        ],
    },
    url="https://github.com/RubenBranco/PatientFlow",
)
