from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name="eval",  # Name of your package
    version="0.1",  
    description="RAG evaluation framework",  # Short description
    packages=find_packages(),  # Automatically find packages in the folder
    install_requires=install_requires,
    # extras_require={
    #     "dev": [
    #         "pytest",
    #         "twine",
    #     ]
    # },
    # classifiers=[
    #     "Programming Language :: Python :: 3.6",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    python_requires='>=3.12',  # Python version requirement
)