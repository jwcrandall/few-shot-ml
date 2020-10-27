from setuptools import setup

package_structure = [
    'utils',
    'utils/practical_2'
]

requirements = [
    'jupyterlab',
    'flake8'
]

setup(
    version='1.0.0',
    name='utils',
    packages=package_structure,
    install_requires=requirements,
)
