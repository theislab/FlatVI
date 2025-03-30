from setuptools import find_packages, setup

setup(
    name='scCFM',
    version='1.0.0',
    author='Leon Hetzel, Alessandro Palma, Sergei Rybakov',
    url='',
    packages=['scCFM/']+find_packages(),
    zip_safe=False,
    include_package_data=True
)