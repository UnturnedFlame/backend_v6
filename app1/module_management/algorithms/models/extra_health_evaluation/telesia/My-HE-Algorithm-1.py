import pickle
import ast
import numpy as np
import argparse
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from sklearn.preprocessing import normalize
import networkx as nx

import os


def weights_Barplot(data, save_path, time_key_list, fre_key_list, primary_key_list):
    namelist = [time_key_list, fre_key_list]
    num_groups = len(data)  # 组的数量（横坐标的数量）
    max_num_bars = max(len(group) for group in data)  # 每组中的最大柱子数量
    indices = np.arange(num_groups)  # 横坐标的范围
    bar_width = 0.8 / max_num_bars  # 每个柱子的宽度
    fig, ax = plt.subplots(figsize=(20, 10))

    for i in range(max_num_bars):
        # 提取每组中第 i 个柱子的值，如果该组中没有第 i 个值，则设置为0
        values = [U[i] if i < len(U) else 0 for U in data]
        # 计算每个柱子的位置
        positions = indices + i * bar_width
        # 绘制柱子
        bars = ax.bar(positions, values, bar_width, label=f'二级指标 {i + 1}')

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height / 2, f'{value:.2f}', ha='center', va='center',
                    fontsize=20)

        for i, (group, names) in enumerate(zip(data, namelist)):
            for j, (value, name) in enumerate(zip(group, names)):
                x_pos = i + j * bar_width
                ax.text(x_pos, value, name, ha='center', va='bottom', fontsize=20, color='blue')
    # 添加标签和图例
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams.update({'font.size': 20})
    ax.set_xlabel('指标', fontsize=20)
    ax.set_ylabel('权重', fontsize=20)
    ax.set_title('权重柱状图', fontsize=20)
    ax.set_xticks(indices + bar_width * (max_num_bars - 1) / 2)
    ax.set_xticklabels(primary_key_list, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    return plt.savefig(save_path + '/Weights_barPlot.png')


def split_rows(second_metric, matrix):
    sub_matrices = []
    start_row = 0
    for rows in second_metric:
        end_row = start_row + rows
        sub_matrix = matrix[start_row:end_row, :]
        sub_matrices.append(sub_matrix)
        start_row = end_row
    return sub_matrices


def getW(criteria, b, max_second_metric):
    W, eigen_list = AHP(criteria, b, max_second_metric).run()
    return W, eigen_list


class AHP:
    def __init__(self, criteria, b, max_second_metric):
        self.RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)
        self.criteria = criteria
        self.b = b
        self.num_criteria = criteria.shape[0]
        self.num_project = max_second_metric

    def cal_weights(self, input_matrix):
        criteria = np.array(input_matrix)
        n, n1 = criteria.shape
        assert n == n1, '"准则重要性矩阵"不是一个方阵'
        for i in range(n):
            for j in range(n):
                if np.abs(criteria[i, j] * criteria[j, i] - 1) > 1e-7:
                    raise ValueError('"准则重要性矩阵"不是反互对称矩阵，请检查位置 ({},{})'.format(i, j))

        eigenvalues, eigenvectors = np.linalg.eig(criteria)

        max_idx = np.argmax(eigenvalues)
        max_eigen = eigenvalues[max_idx].real
        eigen = eigenvectors[:, max_idx].real
        eigen = eigen / eigen.sum()

        if n > 9:
            CR = None
            warnings.warn('无法准确判断一致性')
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n - 1] if self.RI[n - 1] != 0 else 0
        return max_eigen, CR, eigen

    def run(self):
        max_eigen, CR, criteria_eigen = self.cal_weights(self.criteria)

        max_eigen_list, CR_list, eigen_list = [], [], []
        for i in self.b:
            max_eigen, CR, eigen = self.cal_weights(i)
            max_eigen_list.append(max_eigen)
            CR_list.append(CR)
            eigen_list.append(eigen)

        pd_print = pd.DataFrame(eigen_list,
                                index=['一级指标U' + str(i + 1) for i in range(self.num_criteria)],
                                columns=['二级指标' + str(i + 1) for i in range(self.num_project)],
                                )
        pd_print.loc[:, '最大特征值'] = max_eigen_list
        pd_print.loc[:, 'CR'] = CR_list
        pd_print.loc[:, '一致性检验'] = pd_print.loc[:, 'CR'] < 0.1

        return criteria_eigen, eigen_list


def plot_tree(list1, list2_time, list2_freq, save_path):
    # 创建一个有向图
    G = nx.DiGraph()

    # 添加根节点
    root = '状态评估'
    G.add_node(root, level=0)

    # 添加二级节点
    for node in list1:
        G.add_node(node, level=1)
        G.add_edge(root, node)

    # 添加时域指标下的三级节点
    for subnode in list2_time:
        G.add_node(subnode, level=2)
        G.add_edge(list1[0], subnode)

    # 添加频域指标下的三级节点
    for subnode in list2_freq:
        G.add_node(subnode, level=2)
        G.add_edge(list1[1], subnode)

    # 获取每个节点的层次
    levels = nx.get_node_attributes(G, 'level')
    pos = nx.multipartite_layout(G, subset_key="level")

    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题

    # 画图
    plt.figure(figsize=(12, 8))
    node_sizes = [6000 if G.nodes[node]['level'] == 0 else 4000 if G.nodes[node]['level'] == 1 else 3000 for node in
                  G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="skyblue", font_size=18, font_color="black",
            font_weight="bold", edge_color="gray")

    # 调整箭头样式
    edge_labels = {edge: '' for edge in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(save_path + '/TreePlot.png')
    plt.show()


# 层次朴素贝叶斯
def gnb_pred(func_dict, test):
    test = pd.DataFrame(test)
    iset = test.iloc[0, :].tolist()
    means = func_dict['means']
    stds = func_dict['stds']
    # iset = test.tolist()#当前测试实例
    iprob = np.exp(-1 * (iset - means) ** 2 / (stds * 2)) / (np.sqrt(2 * np.pi * stds))  # 正态分布公式
    # prob = 1 #初始化当前实例总概率
    #     for k in range(test.shape[1]): #遍历每个特征
    #         prob *= iprob[k] #特征概率之积即为当前实例概率
    #         cla = prob.index[np.argmax(prob.values)] #返回最大概率的类别
    #     result.append(cla)
    # test['predict']=result
    # acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean() #计算预测准确率
    # print(f'模型预测准确率为{acc}')
    iprob = np.array(iprob).transpose()
    iprob_norm = normalize(iprob, axis=1, norm='l1')
    return iprob_norm


if __name__ == '__main__':

    # 添加命令行参数
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--weight-1', type=list, default=[[1, 3], [1 / 3, 1]])  # 一级指标权重
    parser.add_argument('--weight-2-1', type=list,
                        default=[[1, 3, 1, 5], [1 / 3, 1, 1 / 3, 2], [1, 3, 1, 3], [1 / 5, 1 / 2, 1 / 3, 1]])  # 二级指标权重1
    parser.add_argument('--weight-2-2', type=list, default=[[1, 4, 2, 5], [1 / 4, 1, 1 / 3, 2], [1 / 2, 3, 1, 3],
                                                            [1 / 5, 1 / 2, 1 / 3, 1]])  # 二级指标权重2
    parser.add_argument('--primary-key-list', type=list, default=['时域指标', '频域指标'])  # 一级指标名称
    parser.add_argument('--second-key-list-1', type=list, default=['均方根', '峰峰值', '峰度', '偏度'])  # 二级指标名称
    parser.add_argument('--second-key-list-2', type=list,
                        default=['重心频率', '均方频率', '均方根频率', '频率标准差'])  # 一级指标名称
    parser.add_argument('--statuses-names', type=list, default=['正常', '轻微退化', '严重退化', '失效'])  # 健康状态名称
    parser.add_argument('--suggestion-dict', type=ast.literal_eval, default={
        '正常': '设备当前处于正常工作状态，但为了保持其长期稳定运行，建议定期进行全面检查和清洁保养，尤其注意关键部位的润滑维护。同时，操作人员进行培训，确保其了解正确的操作流程和应急措施。',
        '轻微退化': '设备已经表现出轻微的退化迹象，建议立即进行详细检查，确定退化的具体部位和原因。针对发现的问题进行局部维修或更换磨损零部件，同时加密检查频率，以防止进一步退化并恢复设备的正常运行状态。',
        '严重退化': '设备出现严重退化，建议立即停机进行全面检查和评估。制定详细的维修计划，包括更换损坏或老化的关键零部件，并检查设备的所有连接和系统。维修后进行全面测试，确保设备恢复到正常工作状态，同时考虑加强未来的预防性维护措施。',
        '失效': '设备已完全失效，建议立即停机并进行全面诊断，确定失效原因和范围。与设备制造商或专业维修团队合作，制定和实施详细的修复计划，包括更换损坏的核心部件和系统。修复后进行严格的测试和验收，确保设备完全恢复功能，并制定改进的维护计划以防止类似问题再次发生。'})  # 建议
    parser.add_argument('--save-filepath', type=str, default='results')  # 保存路径
    parser.add_argument('--model-filepath', type=str, default=None)  # 模型参数-'model_1.pkl'
    parser.add_argument('--input-filepath', type=str, default=None)  # 输入数据-'test.mat'

    # 解析命令行参数
    args = parser.parse_args()
    weight_1 = args.weight_1
    weight_2_1 = args.weight_2_1
    weight_2_2 = args.weight_2_2
    primary_key_list = args.primary_key_list
    second_key_list_1 = args.second_key_list_1
    second_key_list_2 = args.second_key_list_2
    statuses_names = args.statuses_names
    suggestion_dict = args.suggestion_dict
    save_filepath = args.save_filepath
    model_filepath = args.model_filepath
    input_filepath = args.input_filepath
    criteria = np.array(weight_1)
    b = [np.array(weight_2_1), np.array(weight_2_2)]

    # 加载健康评估模型
    with open(model_filepath, 'rb') as file:
        model_para = pickle.load(file)
    func_dict = model_para['function_dict']
    second_metric = []
    for i in range(len(b)):
        second_metric.append(b[i].shape[0])  # 每个一级指标下的二级指标个数
    W, W_array = getW(criteria, b, max(second_metric))  # 获取权重矩阵

    # 模型推理
    test_data = np.load(input_filepath)  # 加载输入数据
    test_data = test_data.reshape(1, -1)
    level_matrix = gnb_pred(func_dict, test_data)
    np.save('./level_matrix.npy', level_matrix)

