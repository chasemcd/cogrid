import setuptools

setuptools.setup(
    name="cogrid",
    version="v0.0.1",
    description="Multi-agent extension and expansion of MiniGrid.",
    author="Chase McDonald",
    author_email="chasemcd@cmu.edu",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "gymnasium==0.29.1",
        "pettingzoo==1.24.3",
        "pygame==2.5.2",
    ],
)
