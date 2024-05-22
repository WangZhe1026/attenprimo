## Todo

-  思考该把模块放在哪个位置：应该是直接放在attnfmaps里
-  新加模块的输入输出都是什么：decoder，输出是一个batch的数据（还没想好怎么输入），输出是调整后的batch的数据（同样没想好怎么调整），loss：输入是（什么），输出是loss，计算的是形变能量作为loss
-  sanmodel该新返回什么，或者说怎么调整参数的返回值(x)
-  怎么把新的loss加进去,加在什么位置，什么时候计算loss
-  
## 新的todo
1. 首先，新开了一个model，毕竟直接在sanmodel里写有点麻烦，而且用的diffusion net还有点不一样
2. 可能需要重新写一下diffusion net？因为是需要输入原始的点云的好像，所以要思考一下怎么改
3. 可能要用到faust中的_load_mesh函数

## Data

The data and pretrained models can be found [here](https://1drv.ms/u/s!Alg6Vpe53dEDgbgRZB61zfVdUmd1jg?e=3bZWZn).

## Training & Testing

```
python trainer_sup.py run_mode=train run_cfg=exp/log/<<folder_name_with_sup>>/config.yml
python trainer_unsup.py run_mode=train run_cfg=exp/log/<<folder_name_with_unsup>>/config.yml
```

## Use Pretrained Model

```
python trainer_sup.py run_mode=test run_ckpt=exp/log/<<folder_name_with_sup>>/ckpt_latest.pth
python trainer_unsup.py run_mode=test run_ckpt=exp/log/<<folder_name_with_unsup>>/ckpt_latest.pth
```

## Evaluation

```
python eval_corr.py --data_root exp/data --test_roots exp/log/<<folder1_name>> exp/log/<<folder2_name>>...
```

