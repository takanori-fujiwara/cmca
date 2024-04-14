from distutils.core import setup

setup(
    name="cmca",
    version=0.09,
    packages=[""],
    package_dir={"": "."},
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "adjustText",
    ],
    py_modules=["cca", "cmca"],
)
