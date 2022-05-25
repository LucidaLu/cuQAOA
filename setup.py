import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="cuQAOA-LucidaLu",
    version="0.0.1",
    author="LucidaLu",
    author_email="luyiren12@gmail.com",
    description="A CUDA-based QAOA Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LucidaLu/cuQAOA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
    ]
)
