from setuptools import find_packages, setup
from typing import List

# This function will return a list of requirements
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n","") for req in requirements] 

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements

setup(
    name = "ml-project",
    version = "0.0.1",
    author= "Nidhish",
    author_email = "nidhish.kaul@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)