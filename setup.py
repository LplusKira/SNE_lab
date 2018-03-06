from distutils.core import setup
setup(
    name='SNE_lab',
    packages=[
        'SNE_lab',
        'SNE_lab.dataloaders',
        "SNE_lab.poolers",
        "SNE_lab.statevalidators",
        "SNE_lab.updator"
    ],
    version='0.1.4',
    description='Structured Neural Embedding model for research',
    url='https://github.com/LplusKira/SNE_lab',
    author='Po-Kai Chang',
    author_email='pokaichangtwn@gmail.com',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Education",
        "Topic :: Utilities",
    ],
)
