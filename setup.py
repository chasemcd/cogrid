import setuptools

setuptools.setup(
    name="cogrid",
    version="v0.0.1",
    description="Multi-agent extension of MiniGrid.",
    author="Chase McDonald",
    author_email="chasemcd@cmu.edu",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "gymnasium==0.28.1",
        "opencv-python==4.9.0.80",
        "pygame==2.5.2",
        "ray==2.9.2",
    ],
)
