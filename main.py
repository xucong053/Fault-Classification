import os
import tensorflow as tf
import time
import numpy as np
import data_loader
import matplotlib.pyplot as plt
from tqdm import trange


class DNN:
    """DNN模型训练"""

    def __init__(self, path=None, sheet_name="Sheet2",
                 save_model_path="./model/model-1/", batch_size=32,
                 learning_rate=0.001, num_steps=10000, model_dim=(101, 64, 40),
                 steps=10, threshold_value=0.9, dropout=0, save_model_threshold_value=0.8):
        """
        创建模型训练实例
        :param path: 输入xlsx文件路径
        :param save_model_path: 模型保存路径
        :param batch_size: 训练batch大小
        :param num_steps: 迭代步数
        :param steps: 打印评估间隔步数
        :param threshold_value: 输出阈值
        :param dropout: 丢弃率
        :param save_model_threshold_value: 模型保存阈值
        """
        self.path = path
        self.save_model_path = save_model_path
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.steps = steps
        self.all_counts = int(num_steps / steps) - 1
        self.count = 0  # 评估计数
        self.fig_loss = np.zeros([self.all_counts])
        self.fig_acc = np.zeros([self.all_counts])
        self.save_model_threshold_value = save_model_threshold_value  # 保存模型阈值
        self.threshold_value = threshold_value  # 阈值
        self.sheet_name = sheet_name
        self.keep_rate = 1 - dropout  # dropout使神经网络层中随机丢弃神经单元的值的保留率
        self.learning_rate = learning_rate  # 学习率
        self.hidden_dim = model_dim[1:-1]   # 隐藏层维度
        self.input_dim = model_dim[0]   #输入层维度
        self.output_dim = model_dim[-1] #输出层维度

        self.graph = tf.Graph()  # 建立图
        with self.graph.as_default():  # 设为默认的graph
            self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")
            self.train_inputs = tf.placeholder(tf.float32, shape=[None, None], name="train_inputs")  # 占位符，样本输入
            self.train_labels = tf.placeholder(tf.float32, shape=[None, None], name="train_labels")  # 占位符，样本标签
            # 使用GPU进行张量计算
            with tf.device('/cpu:0'):
                # 隐藏层1
                self.hidden_layer = self.layer(output_dim=self.hidden_dim[0], input_dim=self.input_dim,
                                               inputs=self.train_inputs,
                                               keep_prob=self.keep_prob,
                                               activation=None)
                # 隐藏层(>1)
                for num in range(len(self.hidden_dim) - 1):
                    self.hidden_layer = self.layer(output_dim=self.hidden_dim[num + 1], input_dim=self.hidden_dim[num],
                                                   inputs=self.hidden_layer,
                                                   keep_prob=self.keep_prob,
                                                   activation=None)
                # 输出层
                self.output_layer = self.layer(output_dim=self.output_dim, input_dim=self.hidden_dim[-1],
                                               inputs=self.hidden_layer,
                                               keep_prob=None,
                                               activation=None)

                # 最终使用sigmod函数输出预测结果
                self.prediction = tf.nn.sigmoid(self.output_layer)
                tf.identity(self.prediction, "output")

            # 使用sigmoid_cross_entropy作为损失函数
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_layer, labels=self.train_labels))

            # 定义优化损失函数，使用Adam以及预设学习率训练参数
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.init = tf.global_variables_initializer()

    def layer(self, output_dim, input_dim, inputs, keep_prob=None, activation=None):
        """
        神经网络层
        :param output_dim: 输出维度
        :param input_dim: 输入维度
        :param inputs: 输入数据
        :param activation: 采用激活函数
        :param keep_prob: drop_out保持率
        :return: 下一层输入
        """
        # 权重
        W = tf.Variable(tf.random_normal([input_dim, output_dim]))
        # 偏置
        b = tf.Variable(tf.random_normal([1, output_dim]))
        XWb = tf.matmul(inputs, W) + b
        if keep_prob is not None:
            XWb = tf.nn.dropout(XWb, keep_prob=keep_prob)
        if activation is None:
            outputs = XWb
        else:
            outputs = activation(XWb)
        return outputs

    def train(self, data_inputs, data_labels, keep_prob):
        """
        模型训练
        :param data_inputs: 样本输入
        :param data_labels: 样本标签
        :param keep_prob: dropout保留神经元概率
        """
        with tf.Session(graph=self.graph) as session:
            self.init.run()
            if not os.path.exists(self.save_model_path):
                os.mkdir(self.save_model_path)
            # 检查是否已存在模型，有则恢复训练，重新训练新模型需先删除模型
            if os.path.exists(self.save_model_path + "model.ckpt.meta"):
                ckpt = tf.train.latest_checkpoint(self.save_model_path)
                tf.train.Saver().restore(session, ckpt)
            total_loss = 0
            total_acc = 0
            max_acc = 0

            # 模型保存管理
            saver = tf.train.Saver(max_to_keep=2)
            start_time = time.time()
            # 使用tqdm.trange管理进度条输出
            with trange(self.num_steps) as t:
                for step in t:
                    t.set_description("Training")
                    batch_inputs, batch_labels, one_hot_label = next(
                        data_loader.generate_batch(data_inputs, data_labels, batch_size=self.batch_size))
                    feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels,
                                 self.keep_prob: keep_prob}
                    _, loss, output = session.run([self.optimizer, self.loss, self.output_layer], feed_dict)
                    # 阈值判断输出
                    output = [[1 if i > self.threshold_value else 0 for i in j] for j in output]
                    batch_labels = list(batch_labels)
                    for num, value in enumerate(output):
                        if value == list(batch_labels[num]):
                            total_acc = total_acc + (1. / self.batch_size)
                    # 统计损失和准确率
                    total_loss = total_loss + loss

                    # 每steps次进行一次评估输出
                    if step % self.steps == 0 and step != 0:
                        # 计算平均损失
                        average_loss = total_loss / self.steps
                        average_acc = total_acc / self.steps
                        spend_time = time.time() - start_time
                        t.set_postfix(Loss="{:.9f}".format(average_loss), Accuracy="{:.9f}".format(average_acc))
                        self.fig_loss[self.count] = average_loss
                        self.fig_acc[self.count] = average_acc
                        if average_acc > self.save_model_threshold_value:
                            # 如果评估准确率历史最优，保存模型
                            self.save_model_threshold_value = average_acc
                            saver.save(sess=session, save_path=self.save_model_path + "model.ckpt")
                        # 每一次评估后损失与准确率置零
                        total_loss = 0
                        total_acc = 0
                        # 计数+1
                        self.count += 1
                        # 重置时间
                        start_time = time.time()

    def plot(self):
        # 绘制损失与准确率图
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        lns1 = ax1.plot(np.arange(self.all_counts), self.fig_loss, label="Loss")
        lns2 = ax2.plot(np.arange(self.all_counts), self.fig_acc, 'r', label="Accuracy")
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('training loss')
        ax2.set_ylabel('training accuracy')
        # 合并图例
        lns = lns1 + lns2
        labels = ["Loss", "Accuracy"]
        plt.legend(lns, labels, loc=7)
        plt.show()

    def main(self):
        """模型训练"""
        # 读取样本
        data_inputs, data_labels = data_loader.read_data(path=self.path, sheet_name=self.sheet_name, input_num=self.input_dim, label_num=self.output_dim)
        # 训练样本
        self.train(data_inputs, data_labels, self.keep_rate)
        # 绘制loss与acc图
        self.plot()


if __name__ == '__main__':
    # 创建类的实例
    ###########################################参数说明############################################
    # path: 数据输入路径，sheet_name：数据所在表单名， save_model_path: 模型保存路径
    # batch_size: 训练批次大小，learning_rate: 学习率, num_steps: 总迭代步数
    # model_dim为一个元祖，存放整个模型各层的维度[tuple:（输入层维度, 隐藏层1维度, 隐藏层2维度..., 输出层维度)]
    # steps: 每多少步进行评估，threshold_value: 输出结果阈值，dropout: 防过拟合，随机神经单元丢弃率
    # save_model_threshold_value: 模型保存阈值（评估准确率大于阈值保存模型）
    dnn = DNN(path="./data/样本.xlsx", sheet_name="Sheet2", save_model_path="./models/model-1/",
              batch_size=64, learning_rate=0.001, num_steps=20000, model_dim=(101, 64, 51, 40), steps=100,
              threshold_value=0.9, dropout=0.1, save_model_threshold_value=0.8)
    # 程序入口，开始进行训练
    dnn.main()
