from setuptools import setup, find_packages

setup(
    name='mypackage',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='EDSA exaple python package',
    long_description=open('README.md').read(),
    install_requires=['numpy'],
    url = '<https://github.com/<moses-y>/<package-name>',
    author='<Moses Yebei>',
    author_email='<mosesmy@hotmail.com>'

)