"""
A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Get the version
main_ns = {}
ver_file = (here / 'motrack' / 'version.py').read_text()
exec(ver_file, main_ns)

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='motrack',  # Required
    version=main_ns['__version__'],  # Required
    description='Tracking-by-detection (MOT) package',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional
    url='https://github.com/Robotmurlock/Motrack',  # Optional
    author='Momir Adzemovic',  # Optional
    author_email='momir.adzemovic@gmail.com',  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='tracking-by-detection, multi-object-tracking',  # Optional
    packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests', 'scripts']),  # Required
    python_requires='>=3.8, <4',
    install_requires=[
        'hydra-core',
        'matplotlib',
        'numpy',
        'omegaconf',
        'opencv_python',
        'pandas',
        'PyYAML',
        'scipy',
        'tqdm',
        'torch',
        'motrack-motion'
    ],  # Optional
    extras_require={  # Optional
        'yolov8': ['ultralytics'],
        'reid': ['onnxruntime'],
        'motion': ['motrack-motion']
    }
)
