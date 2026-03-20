from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    req_path = Path(__file__).parent / "requirements.txt"
    lines = req_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


setup(
    name="hybrid-rag-pipeline",
    version="0.1.0",
    description="Hybrid RAG pipeline backend and scripts",
    packages=find_packages(include=["backend", "backend.*"]),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.10",
)

