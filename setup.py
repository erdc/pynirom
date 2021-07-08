from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='pynirom',
    version='0.8.0',
    description='Python tools for non-intrusive reduced order modeling',
    long_description=readme,
    author='Sourav Dutta and Peter Rivera-Casillas and Matthew Farthing',
    author_email='sourav.dutta@erdc.dren.mil',
    include_package_data=True,
    packages=find_packages(exclude=('best_models', 'figures','notebooks','data'))
)
