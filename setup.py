from setuptools import setup, find_packages

setup(name='speechpy',
      version='2.2',
      description='The python package for extracting speech features.',
      author='Amirsina Torfi',
      author_email='amirsina.torfi@gmail.com',
      url='https://github.com/astorfi/speechpyn',
      download_url = 'https://github.com/astorfi/speechpy/archive/2.2.zip',
      packages=find_packages(exclude=('tests', 'docs')),
      include_package_data=True,
      install_requires=[
          'scipy',
          'numpy',
      ],
      zip_safe=False)
