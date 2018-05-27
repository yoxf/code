##  环境:

>python 3.6.1 64bit

>pytorch 0.4 64bit

>PyQt5 5.10.1 64bit

## 数据:

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

可以增加训练物品种类，并在main.py 中修改参数，例中训练4类

```python
model.fc = nn.Linear(num_ftrs, 4)

```
# 训练

```python
python main.py
```

# 应用
```python
python app.py
```

命令行运行代码是,去掉代码中的部分注释(if __name__ == '__main__':后的部分)


# GUI
```python
python ui_gui.py
```







