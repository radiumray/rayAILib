#coding:utf-8
from PyQt5 import QtWidgets, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import sys
import cv2
import os

import cv2
import numpy as np
import time
import os
from PIL import Image, ImageDraw, ImageFont
import scipy.misc
import pickle
import datetime
import tensorflow as tf
import glog as log
from glob import glob
import pandas as pd


#图像标记类
class Mark(QtWidgets.QWidget):    
    def __init__(self, imgPath, lablePath, imageOffsetX, imageOffsetY, excelCon):
        super(Mark,self).__init__()

        self.imgPath=imgPath
        self.lablePath=lablePath
        self.imageOffsetX=imageOffsetX
        self.imageOffsetY=imageOffsetY
        self.excelCon=excelCon
        self.mouseOnImgX=0.0
        self.mouseOnImgY=0.0
        self.xPos=0.0
        self.yPos=0.0
        self.curWidth=0.0
        self.curHeight=0.0


        #左上角点100,100， 宽高1000,900， 可自己设置，未利用布局
        self.setGeometry(100,100,1000,900)  
        self.setWindowTitle(u"坐标标注")  #窗口标题
        self.initUI()


    def initUI(self):
        # self.labelR = QtWidgets.QLabel(u'缩放比例:', self)   #label标签
        # self.labelR.move(200, 20)    #label标签坐标
        # self.editR = QtWidgets.QLineEdit(self)    #存放图像缩放的比例值
        # self.editR.move(250,20)      #编辑框坐标

        self.buttonSave = QtWidgets.QPushButton(u"保存坐标到EXCEL", self)  #保存按钮
        self.buttonSave.move(400,20)                #保存按钮坐标
        self.buttonSave.clicked.connect(self.saveButtonClick)  #保存按钮关联的时间

        self.allFiles = QtWidgets.QListWidget(self)     #列表框，显示所有的图像文件
        self.allFiles.move(10,40)                #列表框坐标
        self.allFiles.resize(180,700)            #列表框大小
        allImgs = os.listdir(self.imgPath)            #遍历路径，将所有文件放到列表框中

        allImgs.sort(key= lambda x:int(x[:-4])) #按文件名大小排序

        for imgTmp in allImgs:
            self.allFiles.addItem(imgTmp)
        
        imgNum=self.allFiles.count()

        self.labelShowNum = QtWidgets.QLabel(u'图片数量:'+str(imgNum), self)   #label标签
        self.labelShowNum.move(20, 20)    #label标签坐标


        self.allFiles.itemClicked.connect(self.itemClick)   #列表框关联时间，用信号槽的写法方式不起作用
        self.allFiles.itemSelectionChanged.connect(self.itemSeleChange)

        self.labelImg = QtWidgets.QLabel("选中显示图片", self)  # 显示图像的标签
        self.labelImg.move(self.imageOffsetX, self.imageOffsetY)                        #显示图像标签坐标
        

    # def closeEvent(self, event):

    #     self.file.close()
    #     print('file close')
    #     # event.ignore()  # 忽略关闭事件
    #     # self.hide()  # 隐藏窗体


    # cv2img转换Qimage
    def img2pixmap(self, image):
        Y, X = image.shape[:2]
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order='C')
        self._bgra[..., 0] = image[..., 0]
        self._bgra[..., 1] = image[..., 1]
        self._bgra[..., 2] = image[..., 2]
        qimage = QImage(self._bgra.data, X, Y, QImage.Format_RGB32)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    # 选择图像列表得到图片和路径
    def selectItemGetImg(self):
        imgName=self.allFiles.currentItem().text()
        imgDirName = self.imgPath + self.allFiles.currentItem().text()  #图像的绝对路径
        imgOri = cv2.imread(str(imgDirName),1)      #读取图像
        self.curHeight = imgOri.shape[0]             #图像高度
        self.curWidth = imgOri.shape[1]           # 计算图像宽度，缩放图像
        return imgOri, imgName

    # 显示坐标和图片
    def pointorShow(self, img, x, y):
        cv2.circle(img,(x, y),3,(0,0,255),2)
        cv2.circle(img,(x, y),5,(0,255,0),2)
        self.labelImg.resize(self.curWidth,self.curHeight)                     #显示图像标签大小，图像按照宽或高缩放到这个尺度
        self.labelImg.setPixmap(self.img2pixmap(img))

    #鼠标单击事件
    def mousePressEvent(self, QMouseEvent):     
        pointT = QMouseEvent.pos()             # 获得鼠标点击处的坐标
        self.mouseOnImgX=pointT.x()-200
        self.mouseOnImgY=pointT.y()-70
        imgOri, _=self.selectItemGetImg()
        self.pointorShow(imgOri, self.mouseOnImgX, self.mouseOnImgY)
        # 保存标签
        self.saveLabelBySelectItem()

    # 列表改变显示图片坐标
    def itemSelectShowImg(self):
        imgOri, imgName=self.selectItemGetImg()
        # 从excel表中得到x,y坐标
        xScal, yScal = self.excelCon.getXYPoint('imageName', imgName)
        # 通过归一化x,y计算真实坐标
        self.mouseOnImgX=int(xScal*self.curWidth)
        self.mouseOnImgY=int(yScal*self.curHeight)
        self.pointorShow(imgOri, self.mouseOnImgX, self.mouseOnImgY)


    def itemClick(self):  #列表框单击事件
        self.itemSelectShowImg()

    def itemSeleChange(self): #列表框改变事件
        self.itemSelectShowImg()


    def saveLabelBySelectItem(self):
        curItem=self.allFiles.currentItem()
        if(curItem==None):
            print('please select a item')
            return
        name=str(curItem.text())
        # 坐标归一化
        self.xPos=self.mouseOnImgX/self.curWidth
        self.yPos=self.mouseOnImgY/self.curHeight
        # 更新或追加记录
        self.excelCon.updateAppendRowBycolName('imageName', name, self.xPos, self.yPos)


    def saveButtonClick(self):   #保存按钮事件
        self.saveLabelBySelectItem()


class imgTools():
    
    def __init__(self):
        self.name = "ray"

    def png2jpg(self, path):
        # path:=>'images/*.png'
        pngs = glob(path)
        for j in pngs:
            img = cv2.imread(j)
            cv2.imwrite(j[:-3] + 'jpg', img)
    
    def txt2Excel(self, txtPathName, excelCon):
        with open(txtPathName, 'r') as f:
            lines = f.readlines()
            imagesNum=len(lines)

            imgNameList=[]
            xList=[]
            yList=[]

            for i in range (imagesNum):
                line=lines[i].strip().split()
                imageName=line[0]
                # 去掉路径
                imageName=imageName[44:]
                print(imageName)
                imgNameList.append(imageName)
                landmark = np.asarray(line[1:197], dtype=np.float32)
                nosice=landmark[54*2:54*2+2]

                xList.append(nosice[0])
                yList.append(nosice[1])
            # 批量追加数据
            colNames=['imageName', 'x', 'y']
            datas=[]
            datas.append(imgNameList)
            datas.append(xList)
            datas.append(yList)

            excelCon.appendRowsAnyway(colNames, datas)

    def CenterLabelHeatMap(self, img_width, img_height, posX, posY, sigma):
        X1 = np.linspace(1, img_width, img_width)
        Y1 = np.linspace(1, img_height, img_height)
        [X, Y] = np.meshgrid(X1, Y1)
        X = X - posX
        Y = Y - posY
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma * sigma
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        return heatmap

    # Compute gaussian kernel
    def CenterGaussianHeatMap(self, img_height, img_width, posX, posY, variance):
        gaussian_map = np.zeros((img_height, img_width))
        for x_p in range(img_width):
            for y_p in range(img_height):
                dist_sq = (x_p - posX) * (x_p - posX) + \
                        (y_p - posY) * (y_p - posY)
                exponent = dist_sq / 2.0 / variance / variance
                gaussian_map[y_p, x_p] = np.exp(-exponent)
        return gaussian_map



class excelTools():
    def __init__(self, lablePath, excelName, sheetName=None):
        self.lablePath = lablePath
        self.excelName=excelName
        self.sheetName=sheetName

    def mkEmptyExecl(self, titleFormat):
        writer = pd.ExcelWriter(self.lablePath+self.excelName, engine='xlsxwriter')
        df=pd.DataFrame(titleFormat)

        # df=pd.DataFrame()
        if(self.sheetName==None):
            df.to_excel(writer, index=False)
        else:
            df.to_excel(writer, sheet_name=self.sheetName, index=False)
        writer.save()

    def updateAppendRowBycolName(self, colName, keyWord, x, y):
        dirName=self.lablePath+self.excelName
        if(self.sheetName==None):
            df = pd.read_excel(dirName)
        else:
            df = pd.read_excel(dirName, sheet_name=self.sheetName)
        value=df.loc[df[colName] == keyWord]
        if(value.empty):
            print('add row at end')
            new=pd.DataFrame({'imageName':[keyWord], 'x':[x], 'y':[y]})
            df=df.append(new,ignore_index=True)
            df.to_excel(dirName, sheet_name=self.sheetName, index=False)
        else:
            print('update x y')
            index=value.index.values[0]
            df.at[index,'x']=x
            df.at[index,'y']=y
            df.to_excel(dirName, sheet_name=self.sheetName, index=False)


    def appendRowsAnyway(self, colNames, datas):
        dirName=self.lablePath+self.excelName
        if(self.sheetName==None):
            df = pd.read_excel(dirName)
        else:
            df = pd.read_excel(dirName, sheet_name=self.sheetName)

        print('add rows at end')

        dataDic={}
        for i in range(len(colNames)):
            dataDic[colNames[i]]=datas[i]

        new=pd.DataFrame(dataDic)
        df=df.append(new,ignore_index=True)
        df.to_excel(dirName, sheet_name=self.sheetName, index=False)


    def searchIndexByColName(self, colName, keyWord):
        dirName=self.lablePath+self.excelName
        if(self.sheetName==None):
            df = pd.read_excel(dirName)
        else:
            df = pd.read_excel(dirName, sheet_name=self.sheetName)
        value=df.loc[df[colName] == keyWord]
        if(value.empty):
            return -1
        else:
            # print('x:',value['x'].values[0])
            return value.index.values[0]

    def getXYPoint(self, colName, keyWord):
        dirName=self.lablePath+self.excelName
        if(self.sheetName==None):
            df = pd.read_excel(dirName)
        else:
            df = pd.read_excel(dirName, sheet_name=self.sheetName)
        value=df.loc[df[colName] == keyWord]
        if(value.empty):
            return -1, -1
        else:
            x=value['x'].values[0]
            y=value['y'].values[0]
            return x,y

    def queryWholeData(self):
        dirName=self.lablePath+self.excelName
        if(self.sheetName==None):
            df = pd.read_excel(dirName)
        else:
            df = pd.read_excel(dirName, sheet_name=self.sheetName)
        data=df.values
        return data