from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE.rst') as f:
    license = f.read()

setup(name='speechpy',
      version='0.1',
      description='The python package for extracting speech features.',
      long_description=readme,
      author='Amirsina Torfi',
      author_email='amirsina.torfi@gmail.com',
      url='https://github.com/astorfi/speech_feature_extraction',
      license=license,
      packages=find_packages(exclude=('tests', 'docs')),
      include_package_data=True,
      install_requires=[
          'scipy',
          'numpy',
      ],
      zip_safe=False)
