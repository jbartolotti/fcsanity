from setuptools import setup, find_packages

setup(
    name="fcsanity",
    version="0.1.0",
    description="Sanity checks for preprocessed fMRI resting state data",
    author="J. Bartolotti",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pandas>=1.1.0",
        "nibabel>=3.2.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
