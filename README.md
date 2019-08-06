# C3AE
#
#[ https://arxiv.org/abs/1904.05059 ] orgin pdf

enviroments:
   numpy, tensorflow(1.8), pandas, feather, opencv
```
    pip install -r requirements.txt
```

download imdb/wiki dataset:
 [ wiki ] ( https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar)
 [ imdb ] ( https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar)
 then extract those data to the "./dataset/"

preprocess:
    >>> python preproccessing/dataset_proc.py

training: 
    >>> python C3AE.py -gpu -p c3ae_v16.h5 -s c3ae_v16.h5 --source wiki 


DETECT: 
   [ mtcnn ] (https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection):  detect and align


origin==paper, our == ourtesting
-------------------------

|origin|wiki(MAE)|imdb(MAE)|
| -- | :--: | :--: |
|  | wiki | 6.57 |
|  | imdb| 6.44 |

-------------------------
|our|wiki|imdb|
| -- | -- | -- |
| v1 | wiki | -- |
| v2 | imdb| -- |
