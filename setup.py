from setuptools import setup

setup(name='speechpy',
      version='0.1',
      description='The funniest joke in the world',
      author='asd',
      author_email='amirsina.torfi@gmail.com',
      license='MIT',
      packages=['speech_feature_extraction'],
      install_requires=[
          'scipy',
          'numpy',
      ],
      zip_safe=False)
