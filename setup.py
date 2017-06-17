from setuptools import setup, find_packages

setup(name='speechpy',
      version='1.0',
      description='The python package for extracting speech features.',
      author='Amirsina Torfi',
      author_email='amirsina.torfi@gmail.com',
      url='https://github.com/astorfi/speech_feature_extraction',
      packages=find_packages(exclude=('tests', 'docs')),
      include_package_data=True,
      install_requires=[
          'scipy',
          'numpy',
      ],
      zip_safe=False)
