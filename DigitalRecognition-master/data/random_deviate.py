import os
import random
import shutil
import cv2


##深度学习过程中，需要划分训练集和验证集、测试集。

# 定义moveFile函数
def moveFile(input_path,output_path):
    pathDir = os.listdir(input_path)  # 取图片的原始路径
    # print(pathDir)
    filenumber = len(pathDir)  # 原文件个数
    rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber个数量的样本图片

    for file_name in sample:
        shutil.move(input_path +'/ '+ file_name, output_path + '/' + file_name)
        return

    #

    if __name__ == '__main__':
        # 此循环为已经划分了0、1标签的文件夹的情况，若无，则删除此循环直接指定输入和输出路径即可
        for m in range(2):
            # 源文件夹路径，根目录
            root = '此处输入根目录'
            # 输入路径为根目录下/train/0,根目录/train/1
            input_path = root + '/train/' + str(m)

            output_path = root + '/val/' + str(m)
            # 检验输出路径是否存在，若不存在则创建
            isExists = os.path.exists(output_path)
            if not isExists:
                os.makedirs(output_path)
            # 执行moveFile命令
            moveFile(input_path,output_path)
