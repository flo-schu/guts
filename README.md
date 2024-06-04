# guts-reduced
Reduced GUTS model (fitted only on survival data) implemented in pymob

## Installation

**If the case study was not already installed as a submodule** with a meta package
like `tktd_nrf2_zfe`, you can install the package as follows.

Open a command line utility (cmd, bash, ...) and execute

```bash
git clone git@github.com:flo-schu/tktd_rna_pulse
cd tktd_rna_pulse
```

Create environment, activate it and install model package. 
```bash
conda create -n guts
conda activate guts
conda install python=3.11
pip install -r requirements.txt
```

download the existing results datasets and unlock the results dataset to be
able to re-compute results. If you mess anything up, just delete the whole dataset
and download again.

```bash
datalad clone git@gin.g-node.org:/flo-schu/tktd_nrf2_zfe__data.git data
datalad clone git@gin.g-node.org:/flo-schu/guts__results.git results
datalad unlock case_studies/guts/results
```

if this is not possible create a new dataset with `datalad create -c text2git results` (see section below)


The case studies should now be ready to use.


## Usage

### Interactive jupyter notebooks 

The preferred files to get started with are in the case_studies directory

- case_studies/guts/scripts/tktd_guts_reduced.ipynb
- case_studies/guts/scripts/tktd_guts_scaled_damage.ipynb
- case_studies/guts/scripts/tktd_guts_full_nrf2.ipynb

Open and interact these files with your preferred development environment. If you
don't have any: `pip install jupyterlab` inside the console with the activated
(conda) environment, navigate to the notebooks and run the notebooks.

### Command line

there is a command line script provided by the `pymob` package which will directly
run inference accroding to the scenario settings.cfg file provided in the scenario
folder of the respective case study. For details see https://pymob.readthedocs.io/en/latest/case_studies.html

`pymob-infer --case_study guts --scenario guts_scaled_damage --inference_backend numpyro`

The options in the `settings.cfg` are the same as used when preparing the publication

The results will be stored in the results directory of the respective case study 
unless otherwise specified in the settings.cfg


## Tracking results and data with datalad and gin

1. create a new `results` dataset in the root of the repository

`datalad create -c text2git results`

if the directory already exists, use the `-f` flag:

`datalad create -f -c text2git results`

2. save the results as they come in in order to keep a version history.

`datalad save -m "Computed posterior with nuts"`


### Upload the dataset to gin.


1. Create a new repository on gin https://gin.g-node.org/

2. add the repository as a remote for the dataset

`datalad siblings add -d . --name gin --url git@gin.g-node.org:/flo-schu/tktd_rna_pulse__results.git`

The remote is now connected to the sibling `gin`. 

3. Push new results to gin `datalad push --to gin` 
