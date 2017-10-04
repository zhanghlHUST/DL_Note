# 交通标志识别算例

## platform 

* win10
* caffe
* python27 (Anaconda2)

## tutorial

* 参见 [准确率98%的深度学习交通标志识别是如何做到的？](http://www.jianshu.com/p/39d1d8af7c57?utm_campaign=haruki&utm_content=note&utm_medium=reader_share&utm_source=weixin)

## get data
 [下载traffic sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

### 数据预览

在 `GTSRB_Final_Training_Images\GTSRB\Final_Training\Images`中共 43 类对象（文件夹），且每种标记的图片数不一致, 约为 `221~2221`, 图片格式为 `.ppm`,可以使用 XnView 预览。且每种对象提供不同分辨率、角度、清晰度和曝光的图片，因此每中对象中有不同的分类

![样本分布](file://C:/Users/10925/Desktop/hist_example.PNG)
![不同类别的预览](file://C:/Users/10925/Desktop/traffic_example.PNG)
![同一类别的不同拍摄条件的预览](file://C:/Users/10925/Desktop/traffic_example2.PNG)

### 图片预处理 

> 需要调整
* 灰度化
* 归一化，[图像预处理](http://blog.csdn.net/fuwenyan/article/details/53899230)认为图片数据默认是在 0~255 之间不需要归一化，《机器学习系统设计》一书中说，减去均值的均一化能够适应不同的光照条件
* 直方图调整等前处理操作，（后续）

### 将图像导入到 lmdb 数据库

>[Caffe中图像写入LMDB和读取LMDB数据](http://blog.csdn.net/langb2014/article/details/52995349) 进行直方图均衡化处理，没有用灰度图

## 网络结构

> ![网络结构](file://C:/Users/10925/Desktop/net_structure.PNG) 

> 精度测试
```
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip3"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
```

## 训练结果

![loss曲线](file://C:/Users/10925/Desktop/loss.png)

## 精度测试

经过 5000 次， batch= 100的优化，测试集精度达到了 99.98 % , 远远超过了参考文献，与参考文献不同的是 这里采用了彩色图像，并采用了直方图均衡技术

**too good to believe**, 本篇文章是我的 caffe 入门文章，对 accuracy 的计算还有疑问，如果有错误，欢迎指出
## github traffic-signs-master 的解决方案

### 参考

* [caffe中特殊的层](http://blog.csdn.net/happynear/article/details/48547383)
* [traffic-signs @ github](https://github.com/navoshta/traffic-signs)

> The highlights of this solution would be data preprocessing, data augmentation, pre-training and skipping connections in the network( By Author himself).
> 解决方案的亮点在于 数据预处理、数据增强、与训练、和跨层连接

### 网络结构

 ![网络结构](file://C:/Users/10925/Desktop/net_structure2.PNG) 

经过 1000 次， batch= 100的优化，测试集精度达到了 98.42 % , 时间问题，需要进一步比较