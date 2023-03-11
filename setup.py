from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='multistack_registration',
    version='1.0.0',
    author='Matthias Walle',
    author_email='matthias.walle@hest.ethz.ch',
    description='Multistack registration for HR-pQCT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/wallematthias/multistack_registration',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'mycmd=multistack_registration.main:main_function',
        ],
    },
)
