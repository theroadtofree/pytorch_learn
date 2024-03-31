#tensorboard --logdir=runs
import numpy as np
import random
from matplotlib import pyplot as plt

# 创建训练数据----------------------------------------------------------------01
w = 2
b = 3
x_train = np.random.randint(low=-10, high=10, size=30)
y_target = [w * x + b + random.randint(0,2) for x in x_train]
# plt.plot(x_train, y_target,'bo')


#  定义模型结果部分----------------------------------------------------------02
import torch
from torch import nn
class LinearModel(nn.Module):
  def __init__(self):                            # 模型的参数放在初始化中进行生成
    super().__init__()
    self.weight = nn.Parameter(torch.randn(1))
    self.bias = nn.Parameter(torch.randn(1))

  def forward(self, input):                      # 模型的计算/卷积，是在前向传播方法中实现
    return (input * self.weight) + self.bias


# 实列化模型对象------------------------------------------------------------03
model = LinearModel()


# 定义好损失函数-----------------------------------------------------------04
loss_Function=nn.MSELoss()  # 实列化一个损失函数

# 定义优化方法-------------------------------------------------------------05
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)


# 模型训练部分-------------------------------------------------------------06
# 实例化TensorboardX-writer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

y_train = torch.tensor(y_target, dtype=torch.float32)

for i_step in range(1000):
    input = torch.from_numpy(x_train)
    output = model(input)
    loss = nn.MSELoss()(output, y_train)
    model.zero_grad()   # 每次反向传播前，要记得提取清零
    loss.backward()
    optimizer.step()
    
    writer.add_scalar('Loss/train', loss, i_step )   # 调用可视化方法