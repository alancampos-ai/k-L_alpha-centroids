## Journal of Computational and Applied Mathematics

#### <img src="figure/Jcam.png" alt="Journal of Computational and Applied Mathematics" width="16" align="absmiddle"/> &nbsp;Article: &nbsp;&nbsp;<small><strong>A Generalized K-Lα Centroids Algorithm for DT-MRI Segmentation</strong> &nbsp;<a href="https://doi.org/10.1016/j.cam.2026.117584"><img src="https://cdn.simpleicons.org/doi/2dd4bf" alt="DOI" height="14" align="absmiddle"></a>&nbsp;&nbsp;<a href="https://doi.org/10.1016/j.cam.2026.117584"><img src="https://cdn.simpleicons.org/zenodo/2dd4bf" alt="Zenodo" height="14" align="absmiddle"></a>&nbsp;&nbsp;<a href="https://drive.google.com/drive/folders/1YBVxdJ12ujfiddyKyu-Z6pwpmwWGmsDr?usp=drive_link"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/google/google-original.svg" alt="Google Drive" height="14" align="absmiddle"></a></small>

We present a generalized *k*–L<sup>α</sup> centroids algorithm for segmenting diffusion tensor MRI (DT-MRI). The method subsumes classical *k*-means and its variants in both Euclidean and Riemannian geometries by representing cluster prototypes as L<sup>α</sup>-centroids. In DT-MRI, each image element is a symmetric positive definite (SPD) matrix. In full 3D acquisitions, voxels are typically modeled by 3×3 SPD tensors, whereas 2×2 SPD matrices arise only in reduced 2D settings. In this work, all experiments are carried out on 3×3 SPD matrices. The SPD cone is an open, convex subset of the space of symmetric matrices and, under the affine-invariant metric, constitutes a Hadamard manifold; in this setting, weighted centroids are well defined and unique. Experiments on real DT-MRI volumes demonstrate computational feasibility and enable a direct, controlled comparison between Euclidean and Riemannian formulations. We compare EUC/LOG/AIRM formulations and characterize the performance–cost trade-off without claims of general superiority.
  
#### Dependencies 
- Python >= 3.8
- numpy
- scipy
- dipy
- matplotlib
- pymanopt

#### Bash
pip install numpy scipy dipy matplotlib pymanopt

#### License (MIT)
- Copyright (c) alancampos-ai
- Code released under the MIT License.
