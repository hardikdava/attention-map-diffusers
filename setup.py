from setuptools import setup, find_packages

setup(
    name="attention_map_diffusers",
    version="0.1.5",
    author="wooyeolbaek",
    author_email="100wooyeol@gmail.com",
    description="attention map for diffusers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wooyeolBaek/attention-map-diffusers",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "git+https://github.com/huggingface/diffusers.git@56d438727036b0918b30bbe3110c5fe1634ed19d",
        "accelerate",
        "transformers",
        "einops",
        "torchvision",
        "protobuf",
        "sentencepiece",
    ],
)
