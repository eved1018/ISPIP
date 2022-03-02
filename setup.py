import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ISPIP",
    version="0.0.1",
    author="Evan Edelstein",
    author_email="edelsteinevan@gmail.com",
    description="Easy to use meta-method prediction of protein-protein interfaces. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eved1018/ISPIP,
    project_urls={
        "Bug Tracker": "https://github.com/eved1018/ISPIP,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "metadpi"},
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
