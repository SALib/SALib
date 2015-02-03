from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = 'SALib/_version.py'
versioneer.versionfile_build = None
versioneer.tag_prefix = 'v' # tags are like 1.2.0
versioneer.parentdir_prefix = 'SALib-' # dirname like 'myproject-1.2.0'

class NoseTestCommand(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

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
        author="Jon Herman",
        author_email="jdherman8@gmail.com",
        license=open('LICENSE.md').read(),
        tests_require=['nose'],
        install_requires=[
            "numpy>1.7",
            "scipy",
        ],
        # Two arguments required by Versioneer
        version = versioneer.get_version(),
        cmdclass=cmdclass,
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


if __name__ == "__main__":
    setup_package()
