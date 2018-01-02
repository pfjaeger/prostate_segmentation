# nci_isbi2013_segmentation by Paul F. Jaeger
CG and PZ Prostate segementation on T2-weighted MR images.

## How to use it
Specify local data paths in config.py
python preprocessing.py
python exec.py
python exec.py --mode test --exp /path/to/exp/dir 

## How to install locally

Install dependencies
```
pip install numpy scipy nilearn matplotlib scikit-image nibabel
```

Install batchgenerators
```
git clone https://phabricator.mitk.org/source/dldabg.git
cd dldabg
pip install .
```


