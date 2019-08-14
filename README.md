# [C3AE]( https://arxiv.org/abs/1904.05059 )

This is a unofficial keras implements of c3ae for age estimation. welcome to discuss ~ 

## required enviroments:
   numpy, tensorflow(1.8), pandas, feather, opencv, python=2.7
   
   >>> pip install -r requirements.txt

##  Preparation
*download*  imdb/wiki dataset and then *extract* those data to the "./dataset/" \
 [download wiki]( https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar) 
 [download imdb]( https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar)
 

## Preprocess:
    >>>  python preproccessing/dataset_proc.py -i ./dataset/wiki_crop --source wiki
    >>>  python preproccessing/dataset_proc.py -i ./dataset/imdb_crop --source imdb

## training: 
    >>> python C3AE.py -gpu -p c3ae_v16.h5 -s c3ae_v16.h5 --source wiki 


## DETECT: 
   [mtcnn] (https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection):  detect\align\random erasing \
   ![trible box](https://raw.githubusercontent.com/StevenBanama/C3AE/master/assets/triple_boundbox.png)


### origin==paper
-------------------------

|source|dataset|MAE|
| -- | :--: | :--: |
| from papper | wiki | 6.57 |
| from papper | imdb| 6.44 |

### our == Exploring (to do)

|source|dataset|MAE|
| :--: | :--: | :--: |
| v2 | imdb-wiki| 10.2(without pretrain， -_-||) |


## Questions: 
   - only 10 bins in paper: why we got 12 category: we can split it as "[0, 10, ... 110 ]" by two points!\
   -  Conv5 1 * 1 * 32, has 1056 params, which mean 32*32 + 32. It contains a conv(1*1*32) with bias and global pooling.
![params](https://raw.githubusercontent.com/StevenBanama/C3AE/master/assets/params.png)

# puzzlement:
   
