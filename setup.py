import setuptools

setuptools.setup(
    name="cogrid",
    version="v0.0.1",
    description="Multi-agent extension and expansion of MiniGrid.",
    author="Chase McDonald",
    author_email="chasemcd@cmu.edu",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pettingzoo==1.24.3",
        "opencv-python==4.9.0.80",
        "pygame==2.5.2",
    ],
)
