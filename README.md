
SSD-TensorFlow
==============

Overview
--------

The programs in this repository train and use a Single Shot MultiBox Detector
to take an image and draw bounding boxes around objects of certain classes
contained in this image. The network is based on the VGG-16 model and uses
the approach described in [this paper][1] by Wei Liu et al. The software is
generic and easily extendable to any dataset, although I only tried it with
[Pascal VOC][2] so far. All you need to do to introduce a new dataset is to
create a new `source_xxxxxx.py` file defining it.

To train the model on the Pascal VOC data, go to the `pascal-voc` directory
and put images & annotations in train_set & valid_set folders.Do not change
directory names unless needed:

├── pascal-voc
│   └── trainval
│       └── VOCdevkit
│           ├── train_set
│           │   ├── Annotations
│           │   ├── ImageSets
│           │   └── JPEGImages
│           └── valid_set
│               ├── Annotations
│               ├── ImageSets
│               └── JPEGImages

You then need to preprocess the dataset before you can train the model on it.
It's OK to use the default settings, but if you want something more, it's always
good to try the `--help` parameter.

    python process_dataset.py

You can then train the whole thing. It will take around 1000 to get
good results. Again, you can try `--help` if you want to do something custom.

    python train.py

You can annotate images, dump raw predictions, print the AP stats, or export the
results in the Pascal VOC compatible format using the inference script.

    python infer.py --help

for example, 
CUDA_VISIBLE_DEVICES=-1 python infer.py --checkpoint -1 --annotate TRUE --data-source pascal_voc

[1]: https://arxiv.org/pdf/1512.02325.pdf
[2]: http://host.robots.ox.ac.uk/pascal/VOC/
[3]: http://host.robots.ox.ac.uk:8080/anonymous/NEIZIN.html
[4]: http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
[5]: http://host.robots.ox.ac.uk:8080/anonymous/FYP60C.html
