from setuptools import find_packages, setup

setup(
    name='flatvi',
    version='1.0.0',
    author='Alessandro Palma, Sergei Rybakov, Leon Hetzel',
    url='',
    packages=['flatvi/']+find_packages(),
    zip_safe=False,
    include_package_data=True
)
