from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

import versioneer
setup(version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass())


class NoseTestCommand(TestCommand):

    def run_tests(self):
        # Run nose ensuring that argv simulates running nosetests directly
        import nose
        nose.run_exit(argv=['nosetests'])


def setup_package():
    # Assemble additional setup commands
    cmdclass = versioneer.get_cmdclass()
    cmdclass['test'] = NoseTestCommand

    setup(
        name='SALib',
        packages=find_packages(exclude=["*tests*"]),
        author="Jon Herman and Will Usher",
        author_email="jdherman8@gmail.com",
        license=open('LICENSE.md').read(),
        tests_require=['nose'],
        install_requires=[
            "numpy>=1.9.0",
            "scipy",
            "matplotlib>=1.4.3",
        ],

        extras_require = {
                          "gurobipy": ["gurobipy",]
                          },

        # Two arguments required by Versioneer
        version = versioneer.get_version(),
        cmdclass=cmdclass,
        url="https://github.com/SALib/SALib",
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


if __name__ == "__main__":
    setup_package()
