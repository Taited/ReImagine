import os
from setuptools import setup, find_packages


def _read_requirements(path: str) -> list:
    """Parse requirements.txt without pkg_resources (works in pip's isolated build env)."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
install_requires = _read_requirements(requirements_path)

setup(
    name="diffsynth",
    version="1.1.8",
    description="Enjoy the magic of Diffusion models!",
    author="Artiprocher",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_data={"diffsynth": ["tokenizer_configs/**/**/*.*"]},
    python_requires='>=3.6',
)
