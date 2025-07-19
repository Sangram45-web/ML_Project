from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    It ignores any lines that start with '-' or are empty.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n", "") for req in requirements]
        
        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements

setup(
    name="ML PROJECT",
    version="0.0.1",
    author="Sangram swain",
    author_email="sangramswain880@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
    
)