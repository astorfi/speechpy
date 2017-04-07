from setuptools import setup, find_packages

with open('README.rst') as f:
    readme_file = f.read()

with open('LICENSE') as f:
    license_file = f.read()

setup(name='speechpy',
      version='0.1',
      description='The python package for extracting speech features.',
      long_description=readme_file,
      author='Amirsina Torfi',
      author_email='amirsina.torfi@gmail.com',
      license='MIT',
      url='https://github.com/astorfi/speech_feature_extraction',
      license=license_file,
      packages=find_packages(exclude=('tests', 'docs')),
      install_requires=[
          'scipy',
          'numpy',
      ],
      zip_safe=False)
