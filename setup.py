import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setuptools.setup(
    name="msmhelper",
    version="0.1.0",
    description="Helper functions for Markov State Models.",
    long_description=README,
    long_description_content_type="text/markdown",
    keywords="msm helper",
    author="moldyn-nagel",
    author_email="daniel.nagel@physik.uni-freiburg.de",
    url="https://github.com/moldyn/msmhelper",
    license="BSD 3-Clause License",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=setuptools.find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    install_requires=["numpy", "pandas"],
)