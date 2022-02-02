from setuptools import find_packages, setup

setup(
    name="lxh_prediction",
    packages=find_packages(exclude=["data", "scripts", "experiments"]),
    install_requires=["lightgbm", "scikit-learn", "matplotlib", "torch"],
)
