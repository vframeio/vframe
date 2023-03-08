from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

setup(
    name='vframe',
    version='0.1.0',    
    description='VFRAME: Visual Forensics and Metadata Extraction',
    url='https://github.com/vframeio/vframe',
    author='Adam Harvey',
    license='MIT',
    package_dir={'vframe':'src/vframe'},
    packages=find_packages('src'),
    install_requires=[requirements],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License' 
    ],
     entry_points = {
        'console_scripts': ['vf=src.cli:cli'],
    }
)