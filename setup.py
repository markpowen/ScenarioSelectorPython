import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scenarioselector",
    version="0.0.1",
    author="Mark Philip Owen",
    author_email="mpowen03@yahoo.co.uk",
    description="Scenario Selector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markpowen/ScenarioSelectorPython",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License 2.0 (Apache-2.0)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
