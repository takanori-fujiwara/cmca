from distutils.core import setup

setup(name='cmca',
      version=0.03,
      packages=[''],
      package_dir={'': '.'},
      install_requires=['numpy', 'scipy', 'pandas', 'sklearn'],
      py_modules=['cca', 'cmca'])
