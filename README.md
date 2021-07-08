# NIROM using Python

A collection of Python implementations of non-intrusive reduced order modeling techniques. This repo provides modules for
1) linear dimension reduction using Proper Orthogonal Decomposition (POD),
2) evolution of latent-space coefficients using a greedy Radial Basis Function (RBF) interpolation,
3) evolution of latent-space coefficients using a Tensorflow-based implementation of neural ordinary differential equations (NODE), and
4) extracting spatio-temporal coherent structures from snapshot data using Dynamic Mode Decomposition (DMD). Numerical examples are provided using a benchmark CFD problem and real world applications of shallow water flow problems.

For details on each method and the numerical examples please refer to -

[1] S. Dutta, M. W. Farthing, E. Perracchione, G. Savant, and M. Putti, “A greedy non-intrusive reduced order model for shallow water equations,” J. Comput. Phys., vol. 439, p. 110378, 2021. [Journal](https://www.sciencedirect.com/science/article/pii/S0021999121002734?via%3Dihub)
[arXiv](https://arxiv.org/abs/2002.11329)

[2] S. Dutta, P. Rivera-casillas, and M. W. Farthing, “Neural Ordinary Differential Equations for Data-Driven Reduced Order Modeling of Environmental Hydrodynamics,” in Proceedings of the AAAI 2021 Spring Symposium on Combining Artificial Intelligence and Machine Learning with Physical Sciences, 2021. [Proceedings](https://sites.google.com/view/aaai-mlps/proceedings?authuser=0)
[arXiv](https://arxiv.org/abs/2104.13962)

[3] S. Dutta, P. Rivera-Casillas, O. M. Cecil, M. W. Farthing, E. Perracchione, and M. Putti, “Data-driven reduced order modeling of environmental hydrodynamics using deep autoencoders and neural ODEs,” in Proceedings of the IXth International Conference on Computational Methods for Coupled Problems in Science and Engineering (COUPLED PROBLEMS 2021), 2021, pp. 1–16.
[arXiv](https://arxiv.org/abs/2107.02784)


## Getting Started


### Dependencies

* Python 3.x
* Tensorflow TF 2 / 1.15.0 or above. Prefereably TF 2.0+, as the entire tfdiffeq codebase requires Eager Execution. Install either the CPU or the GPU version depending on available resources.
* tfdiffeq - Installation directions are available at [tfdiffeq](https://github.com/titu1994/tfdiffeq).

A list of all the package requirements along with version information is provided in the [requirements](requirements.txt) file.

### Executing program

* NODE scripts, available inside the notebooks directory, can be invoked with various user-specified configuration options to test different NN models.
* DMD and PODRBF notebooks are also available inside the notebooks directory.
* High-fidelity snapshot data files are available for

Shallow Water models - [Link](https://drive.google.com/drive/folders/1yhudg8RPvwV9SJx9CTqANEnyN55Grzem?usp=sharing),

Navier Stokes model - [Link](https://drive.google.com/drive/folders/1QG4dyoil5QGHjx3d1L3t0S6lsTGS7Vh0?usp=sharing).

These data files should be placed in the <node\_nirom/data/> directory.
* Some pre-trained ROM model files are available at [NIROM models](https://drive.google.com/drive/folders/19DEWdoS7Fkh-Cwe7Lbq6pdTdE290gYSS?usp=sharing). The DMD and PODRBF trained models should be placed in the <node\_nirom/data/> directory, and the NODE models should be placed inside the corresponding subdirectory of <node\_nirom/best\_models>.

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
publisher = {International Center forNumerical Methods in Engineering (CIMNE)},
title = {{Data-driven reduced order modeling of environmental hydrodynamics using deep autoencoders and neural ODEs}},
year = {2021}
}

```


## Acknowledgments

* Thank you to ERDC-HPC facilities for support with valuable computational infrastructure
* Thank you to ORISE for support with appointment to the Postgraduate Research Participation Program.

Inspiration, code snippets, etc.
* [tfdiffeq](https://github.com/titu1994/tfdiffeq) for Tensorflow implementation of NODE.
