import setuptools

setuptools.setup(
    name="fast-torch-dataloader",
    version="0.0.1",
    author="IARAI - Institute of Advanced Research in Artificial Intelligence",
    author_email="ivan.svogor@iarai.ac.at, christian.eichenberger@iarai.ac.at, moritz.neun@iarai.ac.at",
    description="Dataloader improvements for faster, parallel data loading",
    url="https://gitlab.lan.iarai.ac.at/division-t/administration/storage-benchmarking",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
