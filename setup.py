from setuptools import setup
from setuptools import find_packages

requirement_list = [r.strip() for r in open('requirements.txt', 'r').readlines() if r]


def main():
    setup(
        name='sdnn',
        install_requires=requirement_list,
        version='0.1a',
        description='Selective Desensitization Neural Network',
        author='Yu Kabasawa',
        packages=find_packages(),
        license='MIT',
        classifiers=[
            'Operating System :: POSIX',
            'Programming Language :: Python :: 3.5'
        ]
    )


if __name__ == '__main__':
    main()
