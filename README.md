# GTP-PNet
Code of GTP-PNet: A residual learning network based on gradient transformation prior for pansharpening.

````
@article{zhang172gtp,
  title={GTP-PNet: A residual learning network based on gradient transformation prior for pansharpening},
  author={Zhang, Hao and Ma, Jiayi},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={172},
  pages={223--239},
  year={2021},
  publisher={Elsevier}
}
````

#### running environment :<br>
python=2.7, tensorflow-gpu=1.9.0.

#### Prepare data :<br>
First, you should construct the training data according to the Wald protocol, and put the training data in "\data\Train_data\......" following the provided examples.

#### To train :<br>
The training process is divided into two stages. In the first stage, please run "CUDA_VISIBLE_DEVICES=0 python train_T.py" to make TNet learn the gradient transformation prior. In the second stage, run "CUDA_VISIBLE_DEVICES=0 python train_P.py" to learn fusing multi-spectral and panchromatic images, in which the trained TNet is used to constrain the preservation of the spatial structures in pansharpening.


#### To test :<br>
Put test images in the "\data\Test_data\......" folders, and then run "CUDA_VISIBLE_DEVICES=0 python test.py" to test the trained P_model.
You can also directly use the trained P_model we provide (Quickbird &  GF-2).
