#from torchsummary import summary

import torch
# conda env export > test.yaml 打包环境用的
#    当我们想再次创建该环境，或根据别人提供的.yaml文件复现环境时，就可以通过下面的命令来复现安装环境了。
#    conda env create -f test.yaml


# 关于测试
# 我感觉应该是 这种？
# python .\myevalerr.py --data_root exp/data --test_roots exp/log/smal_unsup_24-03-23_18-02-51/

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())



