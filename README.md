# Fast-RCNN-Geo

This folder contains an example implementation of Geo Model in Fast-RCNN [1] / MatConvNet.
The Geo Model Fast-RCNN model that was trained on a large synthetic dataset with fractures and breakouts.

There are one entry-point scripts:

* `fast_rcnn_demoGeo.m`: runs the original Caffe model imported in MatConvNet.


Note that the code does ship with a proposal generation method using Selective
Search [2].

The `fast_rcnn_demoGeo.m` code should run out of the box, downloading the
model as needed.

***** Before ***** 

1. git clone MatConvNet version 1.0-beta25 on https://github.com/vlfeat/matconvnet
2. Installing and compiling the library on http://www.vlfeat.org/matconvnet/install/
3. git clone this repository in examples/ 
4. Extract all tar.gz
5. Download Geo Model Fast-RCNN on https://www.dropbox.com/h?preview=net-deployedGeo.mat 

To demo code using the first GPU on your system, use
something like:

    run matlab/vl_setupnn
    addpath examples/fast_rcnnGeo
    fast_rcnn_demoGeo('gpu',1) ; % using GPU
    fast_rcnn_demoGeo ; % using CPU

## References

1.  *Fast R-CNN*, R. Girshick, International Conference on Computer
    Vision (ICCV), 2015.
2.  *Selective Search* Van de Sande, Koen EA, et al. "Segmentation 
    as selective search for object recognition." ICCV. Vol. 1. No. 2. 2011.
