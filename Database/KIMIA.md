# KIMIA DATA introduction

* The KIMIA Path960 is a dataset that was proposed in the following paper:
A Comparative Study of CNN, BoVW and LBP for Classification of Histopathological Images
Meghana Dinesh Kumar, Morteza Babaie, Shujin Zhu, Shivam Kalra, and H.R.Tizhoosh; The 2017 IEEE Symposium Series on Computational Intelligence (IEEE SSCI 2017), Honolulu, Hawaii, USA from Nov. 27 to Dec 1, 2017.

* and we can download this from kaggle for free. https://www.kaggle.com/ambarish/kimia-path-960

* This paper introduced a new dataset of histopathology images "KIMIA Path960". From a collection of more than 400 whole slide images (WSIs) of muscle, epithelial and connective tissue etc., we selected 20 scans that "visually" represented different texture/pattern types (purely based on visual clues). We manually selected 48 regions of interest of same size from each WSI and downsampled them to 308x168 patches. Hence, we obtained a dataset of 960(=20x48) images. The images are saved as color TIF files although we do not use the color information (i.e., the effect of staining) in our experiments.


<img src="/KIMIA_Path_960.png" alt="KIMIA_Path_960" style="zoom:30%;" />
