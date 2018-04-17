# V-GAN in Tensorflow
This repository is Tensorflow implementation of [Retinal Vessel Segmentation in Fundoscopic Images with Generative Adversarial Networks](https://arxiv.org/pdf/1706.09318.pdf). The referenced keras code can be found [here](https://bitbucket.org/woalsdnd/v-gan/downloads/).

![figure01](https://user-images.githubusercontent.com/37034031/38225319-55f0c47c-372f-11e8-839d-a544b06edfc0.png)

## Improvements Compared to Keras Code
1. Data augmentation is changed from off-line to online process, it solved memory limitation problem but it will slow down the training
2. Add train_interval FLAGS to control training iterations between generator and discriminator, for normal GAN train_interval is 1
3. The best model is saved based on the sum of the AUC_PR and AUC_ROC on validation data
4. Add sampling function to check generated results to know what's going on
5. Measurements are plotted on tensorboard in training process
6. The code is written more structurally  
*Area Under the Curve* (AUC), *Precision and Recall* (PR), *Receiver Operating Characteristic* (ROC)

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

### Arguments
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
**Note:** Copy predicted vessel images to the ./results/\[DRIVE|STARE\]/V-GAN folder  
```
python evaluation.py
```
Results are generated in **evaluation** folder. Hierarchy of the folder is  
```
.
├── DRIVE
│   ├── comparison
│   ├── measures
│   └── vessels
└── STARE
    ├── comparison
    ├── measures
    └── vessels
```
**comparison:** difference maps between V-GAN and gold standard  
**measures:** AUC_ROC and AUC_PR curves  
**vessels:** vessels superimposed on segmented masks  
*Area Under the Curve* (AUC), *Precision and Recall* (PR), *Receiver Operating Characteristic* (ROC)  

### DRIVE Results
![picture1](https://user-images.githubusercontent.com/37034031/38852786-1a271f2c-4256-11e8-8907-477bb298cc30.png)  

### STARE Results
![picture2](https://user-images.githubusercontent.com/37034031/38852814-385daf6a-4256-11e8-918e-1301d3a788b0.png)

### DRIVE Dataset
| train_interval |         Model       |  AUC_ROC |  AUC_PR  | Dice_coeff |
|      :---:     |         :---:       |   :---:  |   :---:  |   :---:    |
|        1       |       Pixel GAN     |  0.9049  |  0.8033  |   0.3020   |
|        1       | Patch GAN-1 (10x10) |  0.9487  |  0.8431  |   0.7469   |
|        1       | Patch GAN-2 (80x80) |  0.9408  |  0.8257  |   0.7478   |
|        1       |       Image GAN     |  0.9280  |  0.8241  |   0.7839   |
|        100     |       Pixel GAN     |  0.9298  |  0.8228  |   0.7766   |
|        100     | Patch GAN-1 (10x10) |  0.9263  |  0.8159  |   0.7319   |
|        100     | patch GAN-2 (80x80) |  0.9312  |  0.8373  |   0.7520   |
|        100     |       Image GAN     |  0.9210  |  0.7883  |   0.7168   |
|        10000   |       Pixel GAN     |  0.9353  |  0.8692  |   0.7928   |
|        10000   | Patch GAN-1 (10x10) |  0.9445  |  0.8680  |   0.7938   |
|        10000   | patch GAN-2 (80x80) |**0.9525**|**0.8752**| **0.7957** |
|        10000   |       Image GAN     |  0.9509  |  0.8537  |   0.7546   |

### STARE Dataset
| train_interval |         Model       |  AUC_ROC |  AUC_PR  | Dice_coeff |
|      :---:     |         :---:       |   :---:  |   :---:  |    :---:   |
|        1       |       Pixel GAN     |  0.9368  |  0.8354  |   0.8063   |
|        1       | Patch GAN-1 (10x10) |  0.9119  |  0.7199  |   0.6607   | 
|        1       | Patch GAN-2 (80x80) |  0.9053  |  0.7998  |   0.7902   | 
|        1       |       Image GAN     |  0.9074  |  0.7452  |   0.7198   | 
|        100     |       Pixel GAN     |  0.8874  |  0.7056  |   0.6616   |
|        100     | Patch GAN-1 (10x10) |  0.8787  |  0.6858  |   0.6432   |
|        100     | patch GAN-2 (80x80) |  0.9306  |  0.8066  |   0.7321   |
|        100     |       Image GAN     |  0.9099  |  0.7785  |   0.7117   |
|        10000   |       Pixel GAN     |  0.9317  |  0.8255  |   0.8107   |
|        10000   | Patch GAN-1 (10x10) |  0.9318  |  0.8378  | **0.8087** |
|        10000   | patch GAN-2 (80x80) |**0.9604**|**0.8600**|   0.7867   |
|        10000   |       Image GAN     |  0.9283  |  0.8395  |   0.8001   |  

**Note:** 
- Set higher training intervals between generator and discriminator, which can boost performance a little bit as paper mentioned. However, the mathematical theory behind this experimental results is not clear.
- The performance of V-GAN Tensorflow implementation has a gap compared with [paper](https://arxiv.org/pdf/1706.09318.pdf). Without fully fine-tuning and subtle difference in implementations may be the reasons.

## Architectures
- **Generator:**
<p align="right">
  <img src="https://user-images.githubusercontent.com/37034031/38475926-038a1b64-3be6-11e8-98c7-4b4fbc8d140f.png" height="549" width="748">  
</p>  

- **Discriminator(Pixel):**
<p align="center">
  <img src="https://user-images.githubusercontent.com/37034031/38475809-6975ae08-3be5-11e8-921b-f9e9481fd1b9.png" height="188" width="309">
</p>

- **Discriminator(Patch-1):**
<p align="center">
  <img src="https://user-images.githubusercontent.com/37034031/38476155-36b2a19a-3be7-11e8-8a43-45bb1255c7c3.png" height="598" width="303">
</p>

- **Discriminator(Patch-2):**
<p align="center">
  <img src="https://user-images.githubusercontent.com/37034031/38476206-7ef270ca-3be7-11e8-82f3-65b98a214c3a.png" height="399" width="304">
</p>

- **Discriminator(Image):**
<p align="center">
  <img src="https://user-images.githubusercontent.com/37034031/38476272-caaa2440-3be7-11e8-9b8c-124741d109e8.png" height="600" width="305">
</p>

## Tensorboard
*AUC_ROC*, *AUC_PR*, *Dice_Coefficient*, *Accuracy*, *Sensitivity*, and *Specificity* on validation dataset during training iterations  
- **AUC_ROC:**
<p align="center">
  <img src="https://user-images.githubusercontent.com/37034031/38844972-a47e2a92-4230-11e8-8eaf-48111e915046.png">
</p>

- **AUC_PR:**
<p align="center">
  <img src="https://user-images.githubusercontent.com/37034031/38845022-ef619b3e-4230-11e8-8cd4-10c3b1999c7c.png">
</p>

- **Dice_Coeffcient:**
<p align="center">
  <img src="https://user-images.githubusercontent.com/37034031/38845105-3d37222a-4231-11e8-9110-43560ff1f77d.png">
</p>

- **Accuracy:**
<p align="center">
  <img src="https://user-images.githubusercontent.com/37034031/38845129-5175d10a-4231-11e8-86d7-aec166107491.png">
</p>

- **Sensitivity:**
<p align="center">
  <img src="https://user-images.githubusercontent.com/37034031/38845151-63e3b7f8-4231-11e8-9a56-bffbcf90550f.png">
</p>

- **Specificity:**
<p align="center">
  <img src="https://user-images.githubusercontent.com/37034031/38845594-1e7f9400-4233-11e8-8e1f-ce4022833ea2.png">
</p>

  
