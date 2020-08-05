from distutils.core import setup

setup(name='cmca',
      version=0.01,
      packages=[''],
      package_dir={'': './cmca'},
      install_requires=['numpy', 'scipy', 'pandas', 'sklearn'],
      py_modules=['cca', 'cmca'])
