import tensorflow as tf
import time
import os
import data_loader
import array
import pandas as pd
from tqdm import trange


class Predict:
    """调用模型进行预测"""

    def __init__(self, model_path="./model/", threshold_value=0.9):
        """
        :param model_path: 模型存放路径
        """
        self.detection_graph = tf.Graph()
        self.detection_graph.as_default()
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(model_path + 'model.ckpt.meta')
        self.graph = tf.get_default_graph()
        self.input = self.graph.get_tensor_by_name('train_inputs:0')
        self.label = self.graph.get_tensor_by_name('train_labels:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
        self.Y = self.graph.get_tensor_by_name('output:0')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path))
        self.data = []
        self.threshold_value = threshold_value

    def predict(self, inputs):
        """预测"""
        # self.keep_prob: 1 预测过程不随机丢弃神经单元
        feed_dict = {self.input: inputs, self.keep_prob: 1}
        try:
            prediction = self.sess.run(self.Y, feed_dict)
            # 使用列表推导，将计算结果one_hot化
            prediction = [1 if i > self.threshold_value else 0 for i in prediction[0]]
            self.data.append(prediction)
        except Exception as e:
            print(e)
        return prediction


if __name__ == '__main__':
    # 创建预测类实例
    # model_path: 调用模型路径， threshold_value: 输出结果阈值
    prediction = Predict(model_path="./models/model-1/", threshold_value=0.6)
    # 读取数据
    # input_num: 为输入预测样本的维度
    data_inputs, _ = data_loader.read_data(path="./data/样本_test.xlsx", sheet_name="Sheet1", input_num=101,
                                           mode="Predict")
    count = 0
    with trange(len(data_inputs)) as t:
        for num in t:
            t.set_description("Predicting")
            count += 1
            # 预测
            result = prediction.predict([data_inputs[num]])
            t.set_postfix(Prediction=result)
        # 结果输出到文件中
        data_df = pd.DataFrame(prediction.data)
        data_loader.write_data(prediction.data, path="./data/样本_test.xlsx", sheet_name="Sheet1")
