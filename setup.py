from setuptools import setup, find_packages

try:
    import pypandoc
    l_d = pypandoc.convert_file('README.rst', 'rst')
except ModuleNotFoundError:
    l_d = ''

# Get the version.
version = {}
with open("waterbalans/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='waterbalans',
    version=version['__version__'],
    description='Python Package voor het maken van Waterbalansen bij Waternet',
    long_description=l_d,
    url='http://waternet.nl',
    author='David Brakenhoff',
    author_email='d.brakenhoff@artesia-water.nl, ',
    license='Unknown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Other Audience',
        'Programming Language :: Python :: 3'
    ],
    platforms='Windows, Mac OS-X',
    install_requires=['numpy>=1.19.2',
                      'matplotlib>=3.4.1',
                      'pandas>=1.3.0',
                      'scipy>=0.19',
                      'xmltodict',
                      'hkvfewspy==0.6.2'],
    packages=find_packages(exclude=[]),
    include_package_data=True,
)
