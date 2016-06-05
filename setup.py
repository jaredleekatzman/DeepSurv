from setuptools import setup

with open('README.txt') as file:
    long_description = file.read()

setup(name='deepsurv',
    version='1.0',
    description='Deep Cox Proportional Hazards Network',
    keywords = "survival analysis deep learning cox regression",
    url='https://github.com/jaredleekatzman/DeepSurv',
    author='Jared Katzman',
    author_email='jaredleekatzman@gmail.com',
    license='MIT',
    long_description = long_description,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
    packages=['deepsurv'],
    install_requires=[
        'theano',
        'lasagne',
        'lifelines',
    ],
    # test_suite = 'nose.collector',
    # test_require = ['nose','lasagne','theano']
)
