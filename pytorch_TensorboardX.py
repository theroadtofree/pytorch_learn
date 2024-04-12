'''
python pytorch_TensorboardX.py
tensorboard --logdir=runs
http://localhost:6006/
'''
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 创建一个SummaryWriter的实例
writer = SummaryWriter()

# 使用add_scalar方法
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)       # 自定义名称Loss/train,并可视化监控
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

    
# 使用add_image方法    
# 构建一个100*100，3通道的img数据    
img = np.zeros((3, 100, 100))
img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

writer.add_image('my_image', img, 0)

writer.close()