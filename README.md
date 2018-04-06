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

## Training
Move to **codes** folder and run main.py  
```
python main.py --train_interval=<int> --ratio_gan2seg=<int> --gpu_index=<int> --discriminator=[pixel|patch1|patch2|image] --batch_size=<int> --dataset=[DRIVE|STARE] --is_test=False
```  
- models will be saved in './codes/{}/model\_{}\_{}\_{}'.format(dataset, disriminator, train_interval, batch_size)' folder, e.g., './codes/STARE/model_image_100_1' folder.  
- smapled images will be saved in './codes/{}/sample_\_{}\_{}\_{}'.format(dataset, disriminator, train_interval, batch_size)', e.g., './codes/STARE/sample_image_100_1' folder.  

### arguments
**train_interval:** training interval between discriminator and generator, default: 1    
**ratio_gan2seg:** ratio of gan loss to seg loss, default: 10  
**gpu_index:** gpu index, default: 0  
**discriminator:** type of discriminator [pixel|patch1|patch2|image], default: image  
**batch_size:** batch size, default: 1  
**dataset:** dataset name [DRIVE|STARE], default: STARE  
**is_test:** set mode, default: False  

**learning_rate:** initial learning rate for Adam, default: 2e-4  
**beta1:** momentum term of Adam, default: 0.5  
**iters:** number of iterations, default: 50000  
**print_freq:** print loss information frequency, default: 100  
**eval_freq:** evaluation frequency on validation data, default: 500  
**sample_freq:** sample generated image frequency, default: 200  

**checkpoint_dir:** models are saved here, default: './checkpoints'  
**sample_dir:** sampled images are saved here, default: './sample'  
**test_dir:** test images are saved here, default: './test'  

## Test
```
python main.py --is_test=True --discriminator=[pixel|patch1|patch2|image] --batch_size=<int> --dataset=[DRIVE|STARE]
```
- Outputs of inferece are generated in 'seg_result_{}\_{}\_{}'.format(discriminator, train_interval, batch_size) folder, e.g., './codes/STARE/seg_result_image_100_1' folder.  
- Make sure model already trained with defined dataset, discriminator, training interval, and batch size.

## Evaluation

## Improvements
