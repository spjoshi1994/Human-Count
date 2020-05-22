# ECP5 MobileNetV2-SqDet Human-Count

A network backbone with MobileNetV2 and SqueezeDet detector to Power Human Counting usecase.

## Dataset
  Note that the dataset should be in KITTI format. Images and labels should be arranged as follows:
 - Training Images: 'data_path'/training/images
 - Labels: 'data_path'/training/labels
 - train.txt:'data_path'/ImageSets/train.txt
 - Test Images: 'data_path'/test/img
 - Test Labels: 'data_path'/test/labels





### Setup Environment
```
$ pip install -r requirements.txt
```

### Run Training

1. Configure scripts/train.sh
- Mainly configure dataset path variable: TRAIN_DATA_DIR
- Optionally User can change other configurations.

2. Run command :
```
    $ ./run
```
### Freezing Model
```
$ python src/genpb.py –ckpt_dir <log directory> --freeze
```
