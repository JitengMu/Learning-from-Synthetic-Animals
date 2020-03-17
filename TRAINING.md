# Training Recipe

**Note**
* `--animal` defines animal category, e.g. horse, tiger
* `--dataset` can be `real_animal, synthetic_animal, synthetic_animal_sp, synthetic_animal_multitask`
    * `real_animal:` TigDog dataset with original stacked hourglass augmentation
    * `synthetic_animal:` synthetic animal dataset with original stacked hourglass augmentation
    * `synthetic_animal_sp:` synthetic animal dataset with strong augmentation
    * `synthetic_animal_multitask:` synthetic animal with multitask training (keypoints + part segmentation)
    * `real_animal_sp:` currently only used for the evaluation purpose (note: needs to uncomment line 150 and comment line 151 for training purpose)

#### Example 1: Run the following commands to train and evaluate on real animals(TigDog).

Training:
```
CUDA_VISIBLE_DEVICES=0 python train/train.py --dataset real_animal -a hg --stacks 4 --blocks 1 --image-path ./animal_data/ --checkpoint ./checkpoint/real_animal/horse/horse_hourglass/ --animal horse
```

```
CUDA_VISIBLE_DEVICES=0 python train/train.py --dataset real_animal -a hg --stacks 4 --blocks 1 --image-path ./animal_data/ --checkpoint ./checkpoint/real_animal/horse/horse_hourglass/ --animal horse --resume checkpoint/real_animal/horse/horse_hourglass/model_best.pth.tar --evaluate
```

#### Example 2: Run the following commands to train on synthetic animals and evaluate on real animals(TigDog). 

Training:
```
CUDA_VISIBLE_DEVICES=0 python train/train.py --dataset synthetic_animal_sp -a hg --stacks 4 --blocks 1 --image-path ./animal_data/ --checkpoint ./checkpoint/synthetic_animal/horse/horse_spaug --animal horse
```
Evaluation:
```
CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py --dataset1 synthetic_animal_sp --dataset2 real_animal_sp --arch hg --resume ./checkpoint/synthetic_animal/horse/horse_spaug/model_best.pth.tar --evaluate --animal horse
```

#### Example 3: CC-SSL training on trained synthetic animals (horses) and evaluate on real animals(TigDog)
```
CUDA_VISIBLE_DEVICES=0 python CCSSL/CCSSL.py --num-epochs 60 --checkpoint ./checkpoint/synthetic_animal/horse/horse_ccssl --resume ./checkpoint/synthetic_animal/horse/horse_spaug/model_best.pth.tar --animal horse
```

Evaluation:
```
CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py --dataset1 synthetic_animal_sp --dataset2 real_animal_sp --arch hg --resume ./checkpoint/synthetic_animal/horse/horse_ccssl/synthetic_animal_sp.pth.tar --evaluate --animal horse
```

#### Example 4: Train on synthetic animals (multi-task setting) and evaluate on real animals(TigDog) 

Training:
```
CUDA_VISIBLE_DEVICES=0 python train/train_multitask.py --dataset synthetic_animal_sp_multitask -a hg_multitask --stacks 4 --blocks 1 --image-path ./animal_data/ --checkpoint ./checkpoint/synthetic_animal/horse/horse_multitask --animal horse
```
Evaluation:
```
CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py --dataset1 synthetic_animal_sp_multitask --dataset2 real_animal_sp --arch hg_multitask --resume ./checkpoint/synthetic_animal/horse/horse_multitask/model_best.pth.tar --evaluate --animal horse
```


