import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="example-pkg-soominjeong",
    version="0.0.1",
    author="Soo Min Jeong",
    author_email="soominjeongkr@gmail.com",
    description="Spark Machine Learning Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/soomin-jeong/spark-ml-project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

install_requires=[
       'pyspark==3.0.1'
]
