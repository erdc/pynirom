from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='pynirom',
    version='0.1.0',
    description='Python tools for non-intrusive reduced order modeling',
    long_description=readme,
    author='Sourav Dutta and Peter Rivera-Casillas and Orie Cecil and Matthew Farthing',
    author_email='sourav.dutta@erdc.dren.mil',
    include_package_data=True,
    packages=find_packages(exclude=('figures','examples','data')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6,<3.9'
)
