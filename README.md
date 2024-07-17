# pyNIROM: Non-Intrusive Reduced Order Modeling using Python

A collection of Python implementations of non-intrusive reduced order modeling techniques. This repo provides modules for
1. linear dimension reduction using Proper Orthogonal Decomposition (POD),
2. evolution of latent-space coefficients using a greedy Radial Basis Function (RBF) interpolation,
3. evolution of latent-space coefficients using a Tensorflow-based implementation of neural ordinary differential equations (NODE), and
4. extracting spatio-temporal coherent structures from snapshot data using Dynamic Mode Decomposition (DMD). Numerical examples are provided using a benchmark CFD problem and real world applications of shallow water flow problems.

For details on each method and the numerical examples please refer to -

[1] S. Dutta, M. W. Farthing, E. Perracchione, G. Savant, and M. Putti, “A greedy non-intrusive reduced order model for shallow water equations,” J. Comput. Phys., vol. 439, p. 110378, 2021. [Journal](https://www.sciencedirect.com/science/article/pii/S0021999121002734?via%3Dihub)
[arXiv](https://arxiv.org/abs/2002.11329)

[2] S. Dutta, P. Rivera-casillas, and M. W. Farthing, “Neural Ordinary Differential Equations for Data-Driven Reduced Order Modeling of Environmental Hydrodynamics,” in Proceedings of the AAAI 2021 Spring Symposium on Combining Artificial Intelligence and Machine Learning with Physical Sciences, 2021. [Proceedings](https://sites.google.com/view/aaai-mlps/proceedings?authuser=0)
[arXiv](https://arxiv.org/abs/2104.13962)

[3] S. Dutta, P. Rivera-Casillas, O. M. Cecil, M. W. Farthing, E. Perracchione, and M. Putti, “Data-driven reduced order modeling of environmental hydrodynamics using deep autoencoders and neural ODEs,” in Proceedings of the IXth International Conference on Computational Methods for Coupled Problems in Science and Engineering (COUPLED PROBLEMS 2021), 2021, pp. 1–16.
[arXiv](https://arxiv.org/abs/2107.02784)


## Getting Started


### Dependencies and installation

* Python 3.x
* Tensorflow TF 2 / 1.15.0 or above. Prefereably TF 2.0+, as the entire tfdiffeq codebase requires Eager Execution. Install either the CPU or the GPU version depending on available resources.
* tfdiffeq - Installation directions are available at [tfdiffeq](https://github.com/titu1994/tfdiffeq).

The code has been tested with Python3.7 and Python 3.8, but it should be compatible with any Python3.x. Python2 support is no longer maintained.
A list of all the package requirements is provided in the [requirements](requirements.txt) file.

For installation, it is assumed that Anaconda, pip, and all Nvidia drivers (GPU support) are already installed. Then the following steps can be used to create a conda environment and install all necessary dependencies. Alternatively, python virtual environments can also be used.

1. Create a conda environment: ```conda create -n newenv```
2. Activate conda environment: ```conda activate newenv```
3. Clone the package repository: ```git clone https://github.com/erdc/pynirom.git && cd pynirom```
4. To install the package and its dependencies type: ```./install.sh```

Note: the dependencies can also be manually installed from the appropriate ```requirements``` file using ```pip``` or ```conda``` and then, the ```pynirom``` package can be installed by typing: ```python3 setup.py install```. We recommend following the installation instructions in the official tensorflow documentation at [tensorflow.org](https://www.tensorflow.org/install/pip#linux).


### Example notebooks to get started

The examples directory contains several Jupyter notebooks that demonstrate how to use the different NIROM methodologies to build reduced order models for fluid flow problems governed by the incompressible Navier Stokes equations, the shallow water equations and 2D Burgers' equation.
The high-fidelity snapshot data files are available at
[Shallow Water example](https://drive.google.com/drive/folders/1yhudg8RPvwV9SJx9CTqANEnyN55Grzem?usp=sharing),
[Navier Stokes example](https://drive.google.com/drive/folders/1QG4dyoil5QGHjx3d1L3t0S6lsTGS7Vh0?usp=sharing), and
[2D Burgers' example](https://drive.google.com/drive/folders/1mzIPvazsxNQd_ThjyKJayOJzOy6dYfNb?usp=sharing).
These data files should be placed in the <./data/> directory.

1. The ```PODRBF_cylinder.ipynb``` and ```PODRBF_SW.ipynb``` notebooks demonstrate the use of the PODRBF method for 2D flow around a cylinder and 2D shallow water flows in real world domains respectively.
2. The ```DMD_cylinder.ipynb``` and ```DMD_SW.ipynb``` notebooks illustrate the use of DMD for these problems.
3. The ```PODNODE_cylinder.ipynb```, ```PODNODE_SW.ipynb```, and ```PODNODE_burgers2d``` notebooks illustrate the use of the PODNODE method for the example problems.
4. The ```comparison_cylinder.ipynb``` and ```comparison_SW.ipynb``` notebooks contain visualizations to compare the predictions obtained using the different NIROM methods. To run these notebooks download the pre-computed PODNODE solution files from [Link](https://drive.google.com/drive/folders/19DEWdoS7Fkh-Cwe7Lbq6pdTdE290gYSS?usp=sharing) and place these files according to the instructions in the notebooks. Also, run the DMD and PODRBF notebooks for each example problem to generate the respective NIROM solutions.


## Authors

* **Sourav Dutta** - *Sourav.Dutta@erdc.dren.mil* - ERDC-CHL
* **Matthew Farthing** - *Matthew.W.Farthing@erdc.dren.mil* - ERDC-CHL
* **Peter Rivera-Casillas** - *Peter.G.Rivera-Casillas@erdc.dren.mil* - ERDC-ITL
* **Orie Cecil** - *Orie.M.Cecil@erdc.dren.mil* - ERDC-CHL


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


## Reference

If you found this library useful in your research, please consider citing
```
@article{Dutta_JCP2021,
archivePrefix = {arXiv},
arxivId = {2002.11329},
author = {Dutta, Sourav and Farthing, Matthew W. and Perracchione, Emma and Savant, Gaurav and Putti, Mario},
doi = {10.1016/j.jcp.2021.110378},
eprint = {2002.11329},
journal = {J. Comput. Phys.},
pages = {110378},
publisher = {Elsevier Inc.},
title = {{A greedy non-intrusive reduced order model for shallow water equations}},
volume = {439},
year = {2021}
}

@inproceedings{Dutta_AAAI2021,
title={Neural Ordinary Differential Equations for Data-Driven Reduced Order Modeling of Environmental Hydrodynamics},
author={Dutta, Sourav and Rivera-Casillas, Peter and Farthing, Matthew W.},
booktitle={Proceedings of the AAAI 2021 Spring Symposium on Combining Artificial Intelligence and Machine Learning with Physical Sciences},
archivePrefix = {arXiv},
arxivId = {2104.13962},
eprint = {2104.13962},
year={2021},
publisher={CEUR-WS},
address={Stanford, CA, USA, March 22nd to 24th, 2021},
}

@inproceedings{Dutta_Coupled2021,
address = {Barcelona, Spain},
archivePrefix = {arXiv},
arxivId = {2107.02784},
author = {Dutta, Sourav and Rivera-Casillas, Peter and Cecil, Orie M. and Farthing, Matthew W. and Perracchione, Emma and Putti, Mario},
booktitle = {Proc. IXth Int. Conf. Comput. Methods Coupled Probl. Sci. Eng. (COUPLED Probl. 2021)},
eprint = {2107.02784},
pages = {1--16},
publisher = {International Center for Numerical Methods in Engineering (CIMNE)},
title = {{Data-driven reduced order modeling of environmental hydrodynamics using deep autoencoders and neural ODEs}},
year = {2021}
}

```


## Acknowledgments

* Thank you to ERDC-HPC facilities for support with valuable computational infrastructure.
* Thank you to ORISE for support with appointment to the Postgraduate Research Participation Program.

Inspiration, code snippets, etc.
* [tfdiffeq](https://github.com/titu1994/tfdiffeq) for Tensorflow implementation of NODE.
