import setuptools

setuptools.setup(
    name="cogrid",
    version="0.0.7",
    description="Multi-agent extension and expansion of MiniGrid.",
    author="Chase McDonald",
    author_email="chasemcd@cmu.edu",
    url="https://github.com/chasemcd/cogrid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy==1.26.4",
        "gymnasium==0.29.1",
        "pettingzoo==1.24.3",
    ],
)
