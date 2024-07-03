from setuptools import setup, find_packages

setup(
    name="pseudo",
    version="0.1.0",
    description="A Python package for the PSEUDO heuristic.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Felipe Paradas",
    author_email="felipegparadas@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0,<3.0",
        "python>=3.10"
    ],
    extras_require={
        "dev": [
            "pytest>=8.2,<9.0",
            "jupyter>=1.0,<2.0",
            "pymoo>=0.6,<0.7",
            "ipykernel>=6.29.5,<7.0"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)