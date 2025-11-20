from setuptools import setup, find_packages

setup(
    name="airqc",
    version="0.1.0",
    packages=find_packages(),
    author="Luis Fernando Muñiz Torres",
    author_email="tu_correo@example.com",
    description="Air quality QC/QA and filtering utilities",
    long_description=open("README.txt", encoding="utf8").read(),
    long_description_content_type="text/plain",
    url="https://github.com/tu_usuario/airqc",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
    ],
    python_requires=">=3.8",
)
