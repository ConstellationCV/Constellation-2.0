from setuptools import setup

setup(name='constellationcv',
      version='0.1',
      description='A fast and efficient system for extracting three dimensional information from a two dimensional image',
      url='https://github.com/ConstellationCV/Constellation-2.0/',
      author='Pratham Gandhi, Samuel Schuur',
      author_email='prathamgandhi.school@gmail.com',
      license='MIT',
      packages=['constellationcv'],
      install_requires=[
          'sklearn',
          'numpy',
      ],
      zip_safe=False)
