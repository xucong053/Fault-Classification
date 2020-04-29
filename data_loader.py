import pandas as pd
import random
from openpyxl import load_workbook
import numpy as np
import sys


def read_data(path=None, sheet_name="Sheet1", input_num=101, label_num=40, mode="train"):
    """
    从xlsx文件读取数据
    :param path: 文件路径
    :param sheet_name: 表单名
    :return: 返回样本输入与标签（测试数据标签为空）
    """
    if path is None:
        print("请输入文件路径！")
        sys.exit()
    df = pd.read_excel(path, header=None, sheet_name=sheet_name)  # 读取数据
    df = df.values  # 将数据转换为numpy数组
    # 判断为训练
    # 数据还是测试数据

    if mode is "train":
        assert input_num + label_num == np.shape(df)[1] #读取样本的总维度跟input_num + label_num不一致会报错
        data_inputs = df[:, :input_num]  # 切片，样本输入数据
        data_labels = df[:, input_num:]  # 切片，样本标签数据
    else:
        assert input_num == np.shape(df)[1] #读取样本的维度跟input_num不一致会报错
        data_inputs = df[:, :]  # 切片，样本输入数据
        data_labels = None  # 切片，样本标签数据
    return data_inputs, data_labels


def write_data(data, path=None, sheet_name=1):
    """
    将结果输出到xlsx文件中
    :param path:
    :param sheet_name:
    """
    if path is None:
        print("请输入文件路径！")
        sys.exit()
    df = pd.read_excel(path, header=None, sheet_name=sheet_name)  # 读取数据
    df = df.values  # 将数据转换为numpy数组
    data_df = pd.DataFrame(np.concatenate((df, np.array(data)), axis=1))
    writer = pd.ExcelWriter(path)
    data_df.to_excel(writer, sheet_name=sheet_name, header=None, index=None)
    writer.save()
    print("结果已保存！")


def generate_batch(data_inputs, data_labels, batch_size=32):
    """
    生成训练用的batch样本
    :param data_inputs: 样本输入
    :param data_labels: 样本标签
    :return: 使用迭代器返回训练数据
    """
    # 随机生成batch_size大小的数据
    slice = random.sample(range(len(data_inputs)), batch_size)
    x_batch = data_inputs[slice]
    y_batch = data_labels[slice]
    y_one_hot = slice
    yield x_batch, y_batch, y_one_hot


if __name__ == '__main__':
    read_data(path='data/54.xlsx')
