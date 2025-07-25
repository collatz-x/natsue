from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    Args:
        file_path: str
    Returns:
        List[str]
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        return requirements

setup(
    name='mlproject_creditrisk',
    version='0.1.0',
    author='collatz',
    author_email='collatz.x@proton.me',
    description='Predictive Model for Credit Risk',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)