from setuptools import setup

setup(
    name='Bandits',
    version='1.0.0',
    packages=['Bandits'],
    package_dir={'Bandits': 'src'},
    url='none',
    license='',
    author='KiselevGA',
    author_email='kiselev@isa.ru',
    install_requires=['gym'],
    include_package_data=True
)