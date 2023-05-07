import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gradient-accumulator",
    version="0.4.3",
    author="André Pedersen and David Bouget and Javier Pérez de Frutos and Tor-Arne Schmidt Nordmo",
    author_email="andrped94@gmail.com",
    description="Package for gradient accumulation in TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andreped/GradientAccumulator",
    packages=setuptools.find_packages(exclude=('tests', 'notebooks', 'assets', 'docs', 'shell')),
    install_requires=[
        "tensorflow",
        "numpy<=1.23.2",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.6",
)
