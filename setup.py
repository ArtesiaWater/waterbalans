from setuptools import setup, find_packages

l_d = ''
try:
    import pypandoc

    l_d = pypandoc.convert('README.md', 'rst')
except:
    pass

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
    author='Raoul Collenteur',
    author_email='r.collenteur@artesia-water.nl, ',
    license='Unknown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Other Audience',
        'Programming Language :: Python :: 3'
    ],
    platforms='Windows, Mac OS-X',
    install_requires=['numpy>=1.12', 'matplotlib>=2.0', "pyodbc",
                      'pandas>=0.20', 'scipy>=0.19', 'xmltodict', 'zeep'],
    packages=find_packages(exclude=[]),
    include_package_data=True,
)
