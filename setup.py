from setuptools import setup

setup(
    name='deproject',
    version='1.0',
    description='Utility for deprojecting ESD to density profile.',
    url='',
    author='Kyle Oman',
    author_email='koman@astro.rug.nl',
    license='GNU GPL v3',
    packages=['deproject'],
    install_requires=['numpy', 'scipy', 'rap'],
    include_package_data=True,
    zip_safe=False
)
