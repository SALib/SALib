from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class NoseTestCommand(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # Run nose ensuring that argv simulates running nosetests directly
        import nose
        nose.run_exit(argv=['nosetests'])


setup(
    name='SALib',
    version="0.1",
    packages=find_packages(),
    author="Jon Herman",
    author_email="jdherman8@gmail.com",
    license=open('LICENSE.md').read(),
    tests_require=['nose'],
    cmdclass={'test': NoseTestCommand},
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
