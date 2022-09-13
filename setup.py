from setuptools import setup, find_packages


from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

packages = find_packages()
packages.append('script')

setup(
    name='hsir',
    description="Hyperspectral Image Restoration Toolbox",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Zeqiang-Lai/HSIR',
    packages=packages,
    package_dir={'hsir': 'hsir', 'hsirrun':'hsirrun'},
    version='0.0.1',
    include_package_data=True,
    install_requires=['tqdm', 'qqdm', 'timm'],
)
