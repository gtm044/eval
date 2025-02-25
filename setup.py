from setuptools import setup, find_packages

setup(
    name="src",  # Name of your package
    version="0.1",  
    description="RAG evaluation framework",  # Short description
    # long_description=open("README.md").read(),  # Use README.md for long description
    # long_description_content_type="text/markdown",
    packages=find_packages(),  # Automatically find packages in the folder
    # install_requires=[
    #     "numpy",
    #     "scipy",
    #     "tiktoken",
    #     "transformers",
    #     "accelerate",
    #     "huggingface_hub",  # Example dependencies
    #     "PyPDF2",
    #     "graphing",
    #     "hnswlib"  
    # ],
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