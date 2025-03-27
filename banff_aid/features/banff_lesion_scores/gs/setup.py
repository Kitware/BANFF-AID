"""setup.py."""

from setuptools import find_packages, setup

# setup(
#     name="Glomerulosclerosis",
#     version="0.1.0",
#     packages=find_packages(),
#     install_requires=[
#         # whatever dependencies
#         "numpy",
#         "histomicstk",
#     ],
#     entry_points={
#         "girder_slicer_cli_web": [
#             # Format: "CLIName = your_package.your_module:yourFunction"
#             "Glomerulosclerosis = glom.cli:cliDescription"
#         ],
#     },
# )

setup(
    name="GlomerulosclerosisPlugin",
    # use_scm_version={'local_scheme': prerelease_local_scheme},
    description="Glomerulosclerosis Computation",
    long_description_content_type="text/x-rst",
    author="Austin Allen",
    author_email="austin.allen@kitware.com",
    url="https://github.com/adallen93/banff-lesion-scores",
    packages=find_packages(exclude=["tests", "*_test"]),
    package_dir={
        "banff_lesion_scores": "gs",
    },
    include_package_data=True,
    install_requires=[
        # scientific packages
        # 'nimfa>=1.3.2',
        # 'numpy>=1.21.1',
        # 'scipy>=0.19.0',
        # 'Pillow==9.5.0',
        # 'pandas>=0.19.2',
        # 'imageio>=2.3.0',
        # 'shapely[vectorized]',
        # 'opencv-python-headless<4.7',
        # 'sqlalchemy',
        # 'matplotlib',
        # 'pyvips',
        # 'termcolor',
        # 'seaborn',
        # 'opencv-python',
        # 'scikit-image>=0.19.2',
        # 'scikit-learn==1.0.2',
        # 'lxml==4.8.0',
        # 'joblib==1.1.0',
        # 'tifffile==2023.4.12',
        # 'tiffslide',
        # 'tqdm==4.64.0',
        # 'umap-learn==0.5.3',
        # 'openpyxl',
        # 'xlrd<2',
        # # dask packages
        # 'dask[dataframe]>=1.1.0',
        # 'distributed>=1.21.6',
        # # large image sources
        # 'large-image[sources]',
        # 'girder-slicer-cli-web',
        # 'girder-client',
        # # cli
        # 'ctk-cli',
    ],
    license="Apache Software License 2.0",
    keywords="gs",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    zip_safe=False,
)
