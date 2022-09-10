from setuptools import setup, find_packages

setup(
    name='hsir',
    description="Hyperspectral Image Restoration Toolbox",
    packages=find_packages(),
    version='0.0.1',
    include_package_data=True,
    install_requires=['tqdm', 'qqdm', 'timm'],
)
