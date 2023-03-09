from setuptools import find_packages, setup

setup(
    name='PerturbSeq_CMV',
    version='1.0.0',
    author='Leon Hetzel, Alessandro Palma',
    url='',
    packages=['PerturbSeq_CMV/']+find_packages(),
    zip_safe=False,
    include_package_data=True
)