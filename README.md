# [C3AE]( https://arxiv.org/abs/1904.05059 )

This is a unofficial keras implements of c3ae for age estimation. welcome to discuss ~ 

--------[result]-----------------
<div>
<img src="https://raw.githubusercontent.com/StevenBanama/C3AE/master/assets/example1.jpg" width="200" height="200"><img src="https://raw.githubusercontent.com/StevenBanama/C3AE/master/assets/example2.jpg" width="200" height="200">
</div>

|source|version|IMDB(mae)|WIKI(mae)|
| -- | -- | :--: | :--: |
| from papper | -- | **6.57** | **6.44** |
| our implement | c3ae-v84 | **6.77**(focal_loss) | **6.74** |
| our implement v2 | c3ae-v89 | **6.58**(with SE_NET + focal loss) | -- |
| our implement v3 | c3ae-v90 | **6.53**(with SE_NET + white norm) | -- |



## structs
   - assets 
   - dataset (you`d better put dataset into this dir.)
   - detect (MTCNN and align)
   - download.sh (bash script of downloading dataset)
   - model (pretrain model will be here)
   - nets (all tainging code)
       - C3AE.py 
   - preproccessing (preprocess dataset)
   - tools (has been moved to ./net/test.py)

## Pretain mode(a temp model)
   >> all trainned  model saved in dir named "model"

## required enviroments:
   numpy, tensorflow(1.8), pandas, feather, opencv, python=2.7
   
   >>> pip install -r requirements.txt

## test
 - for image
   >>> python nets/test.py -i assets/timg.jpg
 - for video
   >>> python nets/test.py -v


##  Preparation
*download*  imdb/wiki dataset and then *extract* those data to the "./dataset/" \
 [download wiki]( https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar) 
 [download imdb]( https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar)
 

## Preprocess:
    >>>  python preproccessing/dataset_proc.py -i ./dataset/wiki_crop --source wiki
    >>>  python preproccessing/dataset_proc.py -i ./dataset/imdb_crop --source imdb

## training: 
    >>> python C3AE.py -gpu -p c3ae_v16.h5 -s c3ae_v16.h5 --source imdb -w 10


## DETECT: 
   [mtcnn] (https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection):  detect\align\random erasing \
   ![trible box](https://raw.githubusercontent.com/StevenBanama/C3AE/master/assets/triple_boundbox.png)

## net struct
![ params ](https://raw.githubusercontent.com/StevenBanama/C3AE/master/assets/params.png) ![ plain_model ](https://raw.githubusercontent.com/StevenBanama/C3AE/master/assets/plain_model.png) 


## Questions: 
   - only 10 bins in paper: why we got 12 category: we can split it as "[0, 10, ... 110 ]" by two points!\
   - Conv5 1 * 1 * 32, has 1056 params, which mean 32 * 32 + 32. It contains a conv(1 * 1 * 32) with bias 
   - feat: change [4 * 4 * 32] to [12] with 6156 params.As far as known, it may be compose of  conv(6144+12) ,pooling and softmax.
   - the distribution of imdb and wiki are unbalanced, that1s why change the CE to focal loss

## puzzlement:
  - the result of the feature layer(W2) is far from expected. Maybe our code exists some error.
  
## Reference
  - focal loss: https://github.com/maozezhong/focal_loss_multi_class/blob/master/focal_loss.py
  - mtcnn: https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection
  
