from distutils.core import setup

setup(name='cmca',
      version=0.06,
      packages=[''],
      package_dir={'': '.'},
      install_requires=['numpy', 'scipy', 'pandas', 'sklearn', 'matplotlib'],
      py_modules=['cca', 'cmca'])
