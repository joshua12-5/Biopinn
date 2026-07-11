from setuptools import setup, find_packages

setup(
    name="biopinn",
    version="0.1.0",
    description="Physics-Informed Neural Network platform for nanoparticle drug transport in tumor spheroids",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.11",
)
