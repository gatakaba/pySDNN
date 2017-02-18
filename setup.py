from setuptools import setup
from setuptools import find_packages

requirement_list = [r.strip() for r in open('requirements.txt', 'r').readlines() if r]


def main():
    setup(
        name='pySDNN',
        install_requires=requirement_list,
        version='0.1',
        package_dir={'pySDNN': 'src'},
        description='Selective Desensitization Neural Network implemented in python',
        author='Yu Kabasawa',
        #packages=find_packages(),
        packages=['pySDNN'],
        license='MIT',
        classifiers=[
            'Operating System :: POSIX',
            'Programming Language :: Python :: 3.5'
        ]
    )


if __name__ == '__main__':
    main()
