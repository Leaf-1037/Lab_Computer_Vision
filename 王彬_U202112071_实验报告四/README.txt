./Results文件夹下有Grad-CAM和LayerCAM两个子文件夹，为各个可视化方法的实验结果。

结果文件以*.png图像文件给出，其中命名有：
1）*_category_**_***.png，即以原本分类为*，目标分类为**，使用可视化方法为***的结果
2）*_**_each_channel.png，以原本分类为*，使用可视化方法为**的每一通道结果

代码文件为./test_gradcam.py和./test_layercam.py，分别是Grad-CAM方法和LayerCAM方法进行绘制的代码。