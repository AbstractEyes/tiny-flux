from setuptools import setup, find_packages

setup(
    name="tinyflux",
    version="0.4.2",
    description="Compact diffusion transformer with dual expert distillation",
    author="AbstractPhil",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "diffusers>=0.25",
        "safetensors",
        "accelerate",
        "tqdm",
        "pillow",
        "numpy",
    ],
    extras_require={
        "dev": [
            "tensorboard",
            "torchvision",
        ],
    },
)