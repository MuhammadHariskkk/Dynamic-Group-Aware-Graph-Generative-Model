"""Setup configuration for group_aware_gmm."""

from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    req_file = Path(__file__).resolve().parent / "requirements.txt"
    return [
        line.strip()
        for line in req_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name="groupaware",
    version="0.1.0",
    description="Dynamic Group-Aware Graph Generative Model for ETH/UCY trajectory prediction.",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=read_requirements(),
)
