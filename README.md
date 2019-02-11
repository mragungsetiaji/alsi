Alternating Least Squares Collaborative Filtering for Implicit Dataset Structure
=======

This project provides fast Python implementations of Alternating Least Squares as described in the papers [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) 

Model has multi-threaded training routines, using Cython and OpenMP to fit the models in
parallel among all available CPU cores.  In addition, the ALS model has custom CUDA
kernels - enabling fitting on compatible GPU's. Approximate nearest neighbours libraries such as [Annoy](https://github.com/spotify/annoy), [NMSLIB](https://github.com/searchivarius/nmslib)
and [Faiss](https://github.com/facebookresearch/faiss) 

Basic usage:

```python
import alsi

# initialize a model
model = alsi.als.AlternatingLeastSquares(factors=50)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(item_user_data)

# recommend items for a user
user_items = item_user_data.T.tocsr()
recommendations = model.recommend(userid, user_items)

# find related items
related = model.similar_items(itemid)
```

The examples folder has a program showing how to use this to [compute similar artists on the
last.fm dataset](https://github.com/mragungsetiaji/alsi/blob/master/examples/lastfm.py).

#### Requirements

This library requires SciPy version 0.16 or later. Running on OSX requires an OpenMP compiler,
which can be installed with homebrew: ```brew install gcc```. Running on Windows requires Python
3.5+.

GPU Support requires at least version 8 of the [NVidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). The build will use the ```nvcc``` compiler
that is found on the path, but this can be overriden by setting the CUDAHOME enviroment variable to point to your cuda installation.

This library has been tested with Python 2.7, 3.5, 3.6 and 3.7 on Ubuntu and OSX, and tested with
Python 3.5 and 3.6 on Windows.
