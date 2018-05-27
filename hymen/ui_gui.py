# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import app
import numpy as np
class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(MainWindow)
        
    def setupUi(self, MainWindow):
        
        MainWindow.resize(570, 627)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.horizontalLayout.addWidget(self.lineEdit)
        
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.horizontalLayout.addWidget(self.pushButton)
        
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.verticalLayout_2.addWidget(self.label_3)
        
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.horizontalLayout_2.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.horizontalLayout_2.addWidget(self.label_2)
        
        spacerItem = QtWidgets.QSpacerItem(200, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_3)
        
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.horizontalLayout_2.setStretch(0, 2)
        self.horizontalLayout_2.setStretch(1, 2)
        self.horizontalLayout_2.setStretch(3, 1)
        
        self.verticalLayout.addWidget(self.groupBox_3)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 8)
        self.verticalLayout.setStretch(2, 1)
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 570, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)        
        
        self.pushButton.clicked.connect(self.browse)
        self.lineEdit.setPlaceholderText("请输入图片地址")
        self.pushButton_2.clicked.connect(self.run)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "深度深度"))
        self.groupBox.setTitle(_translate("MainWindow", ""))
        self.pushButton.setText(_translate("MainWindow", "浏览"))
        self.groupBox_2.setTitle(_translate("MainWindow", ""))
        self.groupBox_3.setTitle(_translate("MainWindow", ""))
        self.label.setText(_translate("MainWindow", "识别结果:"))
        self.label_2.setText(_translate("MainWindow", ""))
        self.pushButton_2.setText(_translate("MainWindow", "识别"))

    # 点击浏览按钮，浏览系统查找文件，并把路径读入输入框中
    # 同时，读入图片并显示在图像框中
    def browse(self):
        # 打开资源管理器选取文件
        fileName, filetype = QFileDialog.getOpenFileName(self,  
                                    "选取文件",  
                                    "./",  
                                    "All Files (*);;Text Files (*.txt)")
        # 读取文件路径+文件名并填入lineEdit
        self.lineEdit.setText(str(fileName))
        # 读入文件图片并呈现在界面上
        self.label_3.setPixmap(QtGui.QPixmap(fileName))

    
    # 点击识别按钮，开始运行app.py识别图像，并把结果显示在label中。
    def run(self):
        file = self.lineEdit.text()
#        print("file:",file)
        path_ = "/".join(file.split("/")[:-1])
        name = file.split("/")[-1]
#        print("path:",path_)
#        print("name:",name)
        if path_:
            names = app.get_names(path_)
#            print(names)
            for n in names:
                if name == n:
                    img = app.get_image(path_,[name])
                    img_tensor = app.img_to_tensor(img)
                    img_tensor = img_tensor[np.newaxis,:]
                    predValue = app.application(img_tensor)
#                    print("the image is :", app.classes[predValue[0]])
                    self.label_2.setText(app.classes[predValue[0]])      

if __name__ == "__main__":
    import sys
    app_ = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.run()
    
    #ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app_.exec_())
