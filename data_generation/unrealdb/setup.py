import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="unrealdb",
    version="0.0.1",
    author="Weichao Qiu",
    author_email="qiuwch@gmail.com",
    description="Build a virtual dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiuwch/unrealdb",
    packages=setuptools.find_packages(),
    classifiers=[
    ],
    python_requires='>=3.6',
)
