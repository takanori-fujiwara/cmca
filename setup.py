from distutils.core import setup

setup(name='cmca',
      version=0.08,
      packages=[''],
      package_dir={'': '.'},
      install_requires=[
          'numpy', 'scipy', 'pandas', 'sklearn', 'matplotlib', 'adjustText'
      ],
      py_modules=['cca', 'cmca'])
