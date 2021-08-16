from setuptools import setup, find_packages

packages = find_packages(exclude=("tests", "examples", "doc", "resources",))

setup(name="wehd",
      version="0.1.0",
      description="WEHD, A Weighted Euclidean-Hamming Distance Metric for Heterogeneous Feature Vectors.",
      long_description="WEHD is a heterogeneous distance function for use in scientific Python environments. "
                       "The weights for an optimal metric for a dataset can be discovered using "
                       "gradient-free optimizers, such as Evolution Strategies, in unsupervised "
                       "settings, as demonstrated in this project.",
      license="Apache License 2.0",
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
      ],
      url='http://github.com/lantunes/wehd',
      author="Luis M. Antunes",
      author_email="lantunes@gmail.com",
      packages=packages,
      keywords=["machine learning", "distance metric", "metric learning", "unsupervised metric learning"],
      python_requires=">=3.6",
      install_requires=["numpy >= 1.15.4", "scikit-learn >= 0.24.2"])