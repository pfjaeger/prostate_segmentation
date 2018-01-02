# nci_isbi2013_segmentation by Paul F. Jaeger
CG and PZ Prostate segementation on T2-weighted MR images.


## Get Dependencies

```
pip install numpy scipy tensorflow tensorflow-gpu sklearn matplotlib dicom dicom_numpy pynrrd

git clone https://phabricator.mitk.org/source/dldabg.git
cd dldabg
pip install .
```


## Get the Data
Download the Data set from https://wiki.cancerimagingarchive.net/display/DOI/NCI-ISBI+2013+Challenge%3A+Automated+Segmentation+of+Prostate+Structures

## Execute

Specify you local I/O paths in configs.py

```
python preprocessing.py
python exec.py               #train the network. --folds 0 1 2 3 ... specifies the CV folds to train
python exec.py --mode test --exp /path/to/exp/dir  # get test set predictions and final dice scores
```
