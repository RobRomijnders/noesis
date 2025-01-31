from setuptools import setup, find_packages

setup(name='noesis',
      version='0.1',
      description='',
      url='',
      author='anonymous',
      author_email='anonymous@anonymous.com',
      license='LICENSE.txt',
      install_requires=[
          'numpy',
          'ipykernel',
          'jupyter',
          'scikit-learn',
          'torchtune',
      ],
      packages=find_packages(exclude=('tests')),
      zip_safe=False)
