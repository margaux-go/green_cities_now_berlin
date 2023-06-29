from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='preprocessing',
      version="0.0.1",
      description="Green Cities model preprocessing pipeline",
      license="MIT",
      author="Team green-cities",
      author_email="n",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      include_package_data=True,
      zip_safe=False)
