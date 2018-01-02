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

Specify local data paths and in config.py
```
python preprocessing.py
python exec.py
python exec.py --mode test --exp /path/to/exp/dir 
```
