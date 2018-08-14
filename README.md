# Deep-Relighting
Code for paper "Deep Image-Based Relighting from Optimal Sparse Samples".
For more details, please refer to http://viscomp.ucsd.edu/projects/SIG18Relighting

## Disclaimer
**This software and related data are published for academic and non-commercial use only.**

## Data
The traing and testing data can be downloaded from:
https://drive.google.com/drive/folders/1bmJREP58WNvq9o5nIwNQEEce7cb3uEDk?usp=sharing

* "data/orgTrainingImages" contians the original 512x512 training images rendered using mitsuba.
* "data/train" contains the cropped 128x128 training patches we use for joint training and refinement.
* "data/test" contains the testing dataset rendered using mitsuba.
* "data/real" contains the real data we captured at UC San Diego.
* "shapes" contains the procedually generated shapes, we created to render the dataset.
* "trained/dirs" contains our learnt optimal light directions for different settings.
* "trained/joint" contains our trained networks of different training settings after jointly training.
* "trained/refine" contains our trained networks of different training settings after refining the Relight Net.
* "env" contains a testing environment map.

## Platform
This code is based on python 2.7 with tensorflow 1.3.0. It has been tested on Ubuntu 16.04.

## Usage
* To train the Sample Net and Relight Net jointly with a setting of 90 degree cone and 5 samples. Run:
```
python run_train_joint.py 90 5 -i ../data/train/joint_npy -o OUTFOLDER
```
 
* To refine training a Relight Net with our cropped data. Run:
```
python run_train_refine.py 90 5 -i ../data/train -o OUTFOLDER -j ../trained/joint -d ../trained/dirs
```

* To test a trained Relight Net on our testing dataset. Run:
```
python run_test.py 90 5 -i ../data/test/mapper_100_sameDir -o OUTFOLDER -w ../trained/refine -d ../trained/dirs
```

* To test a trained Relight Net on one of our real data. Run:
```
python run_test.py 90 5 -i ../data/real/0 -o OUTFOLDER -w ../trained/refine -d ../trained/dirs
```

* To render one data under a environment map. Run:
```
python run_renderEnv.py 90 5 -i ../data/real/0 -o OUTFOLDER -w ../trained/refine -d ../trained/dirs
```

* To play with creating new shapes. Run:
```
python run_genShapes.py -o OUTFOLDER
```

For detail settings of the programs, please run them with --help or check the source codes.

## Citation
If you find this work useful for your research, please cite:
```
@article{xu2018deep,
  title={Deep image-based relighting from optimal sparse samples},
  author={Xu, Zexiang and Sunkavalli, Kalyan and Hadap, Sunil and Ramamoorthi, Ravi},
  journal={ACM Transactions on Graphics (TOG)},
  volume={37},
  number={4},
  pages={126},
  year={2018},
  publisher={ACM}
}
```

## Contact
Feal free to contact me if there is any question (zexiangxu@cs.ucsd.edu)
