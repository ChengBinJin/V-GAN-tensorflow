# V-GAN in Tensorflow
This repository is Tensorflow implementation of [Retinal Vessel Segmentation in Fundoscopic Images with Generative Adversarial Networks](https://arxiv.org/pdf/1706.09318.pdf). The referenced keras code can be found [here](https://bitbucket.org/woalsdnd/v-gan/downloads/).

![figure01](https://user-images.githubusercontent.com/37034031/38225319-55f0c47c-372f-11e8-839d-a544b06edfc0.png)

## Package Dependency
- tensorflow 1.16.0
- python 3.5.3
- numpy 1.14.2
- matplotlib 2.0.2
- pillow 5.0.0
- scikit-image 0.13.0
- scikit-learn 0.19.0
- scipy 0.19.0

## Download Data
Original data file strucure was modified for convenience by [Jaemin Son](https://www.vuno.co/team).  
Download data from [here](https://bitbucket.org/woalsdnd/v-gan/src/04e60e8baee6d03721b0d6b0990255bfa115dab6?at=master) and copy data file in the same directory with codes file as following Directory Hierarchy.  

## Directory Hierarchy
```
.
├── codes
│   ├── dataset.py
│   ├── evaluation.py
│   ├── main.py
│   ├── model.py
│   ├── solver.py
│   ├── TensorFlow_utils.py
│   ├── utils.py
├── data
│   ├── DRIVE
│   └── STARE
├── evaluation (get after running evaluation.py)
│   ├── DRIVE
│   └── STARE
├── results
│   ├── DRIVE
│   └── STARE
```
**codes:** source codes  
**data:** original data. File hierarchy is modified for convenience.  
**evaluation:** quantitative and qualitative evaluation. (*get after running evaluation.py*)  
**results:** results of other methods. These image files are retrieved from [here](http://www.vision.ee.ethz.ch/~cvlsegmentation/driu/downloads.html).  
