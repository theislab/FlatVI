scCFM
=======

This is the official repository for the ICML 2025 spotlight paper Enforcing Latent Euclidean Geometry in Single-Cell VAEs for Manifold Interpolation. Many tools for single-cell RNA-seq (scRNA-seq) operate under the assumption that the latent space exhibits approximate Euclidean geometry, using straight lines to estimate cell state transitions and distances. To support and enhance this assumption, we introduce FlatVI, a representation learning model for scRNA-seq data that promotes locally flat geometry in the latent space, making it a natural complement to existing single-cell analysis pipelines.

FlatVI is a Variational Autoencoder (VAE) trained with a negative binomial likelihood tailored to single-cell data, and augmented with geometric regularisation. The VAE's decoder maps latent representations to parameters of a statistical manifold defined by negative binomial distributions. The local geometry of the latent space is governed by the pullback metric, which we regularise toward a scaled identity matrix. This encourages the latent space to adopt a locally Euclidean structure.

Find our work at: 
* [OpenReview](https://openreview.net/forum?id=DoDXFkF10S&referrer=%5Bthe%20profile%20of%20Alessandro%20Palma%5D(%2Fprofile%3Fid%3D~Alessandro_Palma1))
* Soon on ArXiv too


<p align="center">
  <img src="https://github.com/theislab/FlatVI/blob/branch/camera_ready/docs/flatvi.png" ">
</p>

Data availability
------------

All the used datasets and checkpoints will be made publicly available on Zenodo by the time of the conference. Nonetheless, the datasets in this study are public and can be accessed from their original publications. 


Installation
------------

1. Clone our repository 

```
git clone https://github.com/theislab/FlatVI.git
```

2. Create the conda environment:

```
conda env create -f environment.yml
```

3. Activate the environment:

```
conda activate flatvi
```

4. Install the FlatVI package in development mode:

```
cd directory_where_you_have_your_git_repos/FlatVI
pip install -e . 
```

5. Create symlink to the storage folder for experiments:

```
cd directory_where_you_have_your_git_repos/FlatVI
ln -s folder_for_experiment_storage project_folder
```

6. Create `experiment` and `dataset` folder. 

```
cd project_folder
mkdir datasets
mkdir experiments
```

7. Run experiments.
