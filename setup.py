from setuptools import setup
import os

packages = []
root_dir = os.path.dirname(__file__)
if root_dir:
    os.chdir(root_dir)

for dirpath, dirnames, filenames in os.walk('SALib'):
    # Ignore dirnames that start with '.'
    if '__init__.py' in filenames:
        pkg = dirpath.replace(os.path.sep, '.')
        if os.path.altsep:
            pkg = pkg.replace(os.path.altsep, '.')
        packages.append(pkg)

setup(
    name='SALib',
    version="0.1",
    packages=packages,
    author="Jon Herman",
    author_email="jdherman8@gmail.com",
    license=open('LICENSE.md').read(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
    ],
    url="https://github.com/jdherman/SALib",
    long_description=open('README.md').read(),
    description=(
        'Tools for sensitivity analysis. Contains Sobol, Morris, and FAST methods.'),
    # entry_points = {
    #     'console_scripts': [
    #         'salib = SALib.bin.salib:main',
    #     ]
    # },
    classifiers=[
        # 'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        # 'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 2 :: Only',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],)
