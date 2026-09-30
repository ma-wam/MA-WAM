from setuptools import find_packages, setup

setup(
    name="ma-wam",
    description="MA-WAM: test-time world-model planning for multi-agent flow policies.",
    packages=find_packages(
        include=["diffuser", "diffuser.*", "world_model", "world_model.*"]
    ),
)
