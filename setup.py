from setuptools import setup, find_packages

setup(
    name="omsimulator",
    version="0.1.0",
    description="A package for NN-based simulations of IceCube Optical Modules",
    author="Your Name",
    author_email="javivarac98@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "pytorch-lightning",
        "numpy",
        "h5py",
        "healpy",
        "matplotlib",
        "tqdm"
    ],
    python_requires=">=3.7",
)
