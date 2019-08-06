# C3AE
#
# [ papers] ( https://arxiv.org/abs/1904.05059 )

## enviroments:
   numpy, tensorflow(1.8), pandas, feather, opencv\
```
    pip install -r requirements.txt
```

##download imdb/wiki dataset:
 [ wiki ] ( https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar)
 [ imdb ] ( https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar)
 then extract those data to the "./dataset/"

## preprocess:
    >>> python preproccessing/dataset_proc.py

## training: 
    >>> python C3AE.py -gpu -p c3ae_v16.h5 -s c3ae_v16.h5 --source wiki 


## DETECT: 
   [ mtcnn ] (https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection):  detect and align \


origin==paper, our == ourtesting
-------------------------

|origin|wiki(MAE)|imdb(MAE)|
| -- | :--: | :--: |
|  | wiki | 6.57 |
|  | imdb| 6.44 |

-------------------------
|our|wiki|imdb|
| :--: | :--: | :--: |
| v1 | wiki | XXX |
| v2 | imdb| XXX |


![ params ] (https://github.com/StevenBanama/C3AE/tree/master/assets/params.png)
## Questions: 
   - only 10 bins in paper: why we got 12 category: we can split it as "[0, 10, ... 110 ]" by two points!\
   - SE model: we can treat "SE model" as scale factor, but we will be puzzle about the placement.\
        we can find the params of conv5 , "1 * 1 * 32", which has 1056 params. The SE(factor=2) has 1024 params, which means \
        conv5 contains SE and 1X1 conv. 

#puzzlement:
   
