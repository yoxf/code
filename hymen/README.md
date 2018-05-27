##  环境:

python 3.6.1 64bit
pytorch 0.4 64bit
PyQt5 5.10.1 64bit

## 训练数据:

选取类imagenet数据集部分数据,也可以自行收集数据
分类整理,文件按如下方式组织
data/
train/fish/xxx.xx,xxx.xx,...
train/cock/xxx.xx,xxx.xx,...
train/airplant/xxx.xx,xxx.xx,...
train/sailing/xxx.xx,xxx.xx,...

val/fish/xxx.xx,xxx.xx,...
val/cock/xxx.xx,xxx.xx,...
val/airplant/xxx.xx,xxx.xx,...
val/sailing/xxx.xx,xxx.xx,...

test/xxx.xx,xxx.xx,...

可以增加训练物品种类，并在
main.py 中修改参数，例中训练4类
```python
model.fc = nn.Linear(num_ftrs, 4)

```






