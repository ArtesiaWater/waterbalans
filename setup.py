from setuptools import setup, find_packages


with open("README.rst", "r") as fh:
    long_description = fh.read()

# Get the version.
version = {}
with open("waterbalans/version.py") as fp:
    exec(fp.read(), version)

setup(
    name="waterbalans",
    version=version["__version__"],
    description="Python Package voor het maken van Waternet Waterbalansen",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ArtesiaWater/waterbalans",
    author="Davíd Brakenhoff",
    author_email="d.brakenhoff@artesia-water.nl",
    project_urls={
        "Source": "https://github.com/ArtesiaWater/waterbalans",
        "Documentation": "https://waterbalans.readthedocs.io/en/latest/",
        "Tracker": "https://github.com/ArtesiaWater/waterbalans/issues",
        "Help": "https://github.com/ArtesiaWater/waterbalans/discussions",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
    platforms="Windows, Mac OS-X",
    install_requires=[
        "numpy>=1.19.2",
        "matplotlib>=3.4.1",
        "pandas>=1.3.0",
        "scipy>=0.19",
    ],
    packages=find_packages(exclude=[]),
    include_package_data=True,
)
