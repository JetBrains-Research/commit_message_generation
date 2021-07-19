import setuptools

setuptools.setup(
    name="seq2seq_completion",
    packages=setuptools.find_packages(include="seq2seq_completion.*"),
    version="0.0.1",
    license="MIT",
    description="Pipeline for running sequence-to-sequence model for commit messages completion task",
    author="Alexandra Eliseeva",
    author_email="alexandra.eliseeva@jetbrains.com",
    url="",
    download_url="",
    keywords=[],
    python_requires=">=3.8",
    install_requires=["torch==1.7.1", "transformers==4.2.1", "hydra-core==1.0.5"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
