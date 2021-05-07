# Balanced Knowledge Distillation for Long-tailed Learning

Implemention of 

"[Balanced Knowledge Distillation for Long-tailed Learning](http://arxiv.org/abs/2104.10510)"

Shaoyu Zhang, Chen Chen, Xiyuan Hu and Silong Peng
<<<<<<< HEAD
<<<<<<< HEAD

### Requirements 
* Python 3
* [TensorboardX](https://github.com/lanpa/tensorboardX)
* [scikit-learn](https://scikit-learn.org/stable/)

### Data
* Long-tailed CIFAR.

  Follow [LDAM-DRW](https://github.com/kaidic/LDAM-DRW) for long-tailed CIFAR-10/-100 datasets.
* ImageNet-LT, Places-LT and iNaturalist 2018.

  Follow [Classifier-Balancing](https://github.com/facebookresearch/classifier-balancing) for the three large-scale long-tailed datasets. We will update the corresponding code later.
  
### Training
1. Train the teacher model with cross-entropy loss
```
python train_teacher.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE
```

2. Train the student model with BKD loss
```
<<<<<<< HEAD
python train_student.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type BKD --model /path/to/the/teacher/model/
=======
python train_teacher.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type BKD --model /path/to/the/teacher/model/
>>>>>>> master
```

### Acknowledgement
The code is partly based on [LDAM-DRW](https://github.com/kaidic/LDAM-DRW) and [Classifier-Balancing](https://github.com/facebookresearch/classifier-balancing).
=======
>>>>>>> master
=======
>>>>>>> ab51399a7187cb33a52e6e691fb26633548ce296
