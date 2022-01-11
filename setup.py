import setuptools

setuptools.setup(
    name="concurrent-dataloader",
    version="0.0.2",
    author="IARAI - Institute of Advanced Research in Artificial Intelligence",
    author_email="ivan.svogor@iarai.ac.at, christian.eichenberger@iarai.ac.at, moritz.neun@iarai.ac.at",
    description="Dataloader with concurrency improvements",
    url="https://gitlab.lan.iarai.ac.at/division-t/administration/storage-benchmarking",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
