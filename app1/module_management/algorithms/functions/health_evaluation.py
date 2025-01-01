import os.path
import pickle
import subprocess

import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from app1.module_management.algorithms.functions.health_evaluation_train import getLevel, weights_Barplot, split_rows, \
    weights_Barplot_multiple_sensor, gnb_pred, model_pred
from app1.module_management.algorithms.functions.feature_extraction import get_multiple_sensors_example

# test_path = 'datas/test.mat'
# model_path = 'models/model_1.pkl'
# save_path = 'results'


# 绘制层级指标树状图
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

    matplotlib.use('Agg')
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题

    # 画图
    plt.figure(figsize=(20, 10))
    node_sizes = [6000 if G.nodes[node]['level'] == 0 else 4000 if G.nodes[node]['level'] == 1 else 3000 for node in
                  G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="skyblue", font_size=18, font_color="black",
            font_weight="bold", edge_color="gray")

    # 调整箭头样式
    edge_labels = {edge: '' for edge in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(save_path + '/TreePlot.png')
    plt.close()

    return save_path + '/TreePlot.png'


# 绘制传感器指标树状图
def plot_tree_multiple_sensor(list1, list2_sensor1, list2_sensor2, list2_sensor3, save_path):
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
    for subnode in list2_sensor1:
        G.add_node(subnode, level=2)
        G.add_edge(list1[0], subnode)

    # 添加频域指标下的三级节点
    for subnode in list2_sensor2:
        G.add_node(subnode, level=2)
        G.add_edge(list1[1], subnode)

    for subnode in list2_sensor3:
        G.add_node(subnode, level=2)
        G.add_edge(list1[2], subnode)

    # 获取每个节点的层次
    levels = nx.get_node_attributes(G, 'level')
    pos = nx.multipartite_layout(G, subset_key="level")

    matplotlib.use('Agg')
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
    plt.close()

    return save_path + '/TreePlot.png'


# 单传感器数据的健康评估
def model_eval(extracted_features: pd.DataFrame, raw_data_filepath, model_path, save_path, algorithm, extra_algorithm_filepath=None, username=None, multiple_sensors=False):

    if model_path is not None:
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
    elif extra_algorithm_filepath:
        # 构建增值组件的模型路径
        extra_algorithm_name = os.path.basename(extra_algorithm_filepath).split('.')[0]   # 增值组件文件名
        extra_algorithm_dir = os.path.dirname(extra_algorithm_filepath)  # 调用增值组件时，需要指定增值组件所在的目录
        extra_algorithm_user_dir = extra_algorithm_dir + '/' + username  # 不同用户调用增值组件时，每个用户都有自己的文件夹存放数据
        if not os.path.exists(extra_algorithm_user_dir):
            os.makedirs(extra_algorithm_user_dir)
        # 针对专有健康评估算法
        extra_model_path = extra_algorithm_dir + '/' + extra_algorithm_name + '.pkl'
        print(f'extra_model_path: {extra_model_path}')
        # extra_model_path = algorithm_dir + '/' + extra_algorithm + '.pkl'  # 专有算法的评估模型
        with open(extra_model_path, 'rb') as file:
            data = pickle.load(file)
        pass
    else:
        return {'weights_bar': None, 'result_vector': '专有算法文件路径不存在'}
    try:
        # print(f'test_data: {test_data.shape}')
        matplotlib.use('Agg')
        func_dict = data['function_dict']
        W = data['Weight_1']
        W_array = data['Weight_2']
        # U = data['U']
        num_second_level = data['num_second_level']
        status_nums = data['status_nums']

        # 多传感器相比起单传感器所需要的指标较多
        if not multiple_sensors:
            # 单传感器指标
            time_key_list = data['time_key_list']
            fre_key_list = data['fre_key_list']
            test_data = extracted_features
            features_list = time_key_list + fre_key_list
        else:
            # 多传感器指标
            sensor1_key_list = data['sensor1_key_list']
            sensor2_key_list = data['sensor2_key_list']
            sensor3_key_list = data['sensor3_key_list']
            test_data = get_multiple_sensors_example(raw_data_filepath, sensor1_key_list, sensor2_key_list, sensor3_key_list)

        status_names = data['status_names']
        suggestion_dict = data['suggestion_dict']
        primary_key_list = data['primary_key_list']

        num_examples = test_data.shape[0]
        # test_data = feature_extraction.GetTest(test_path, time_key_list, fre_key_list)
        weights_bar_of_all_examples = []
        result_vector_of_all_examples = []
        result_bar_of_all_examples = []
        suggestion_of_all_examples = []
        tree_of_all_examples = []

        num_status = len(status_names)
        result_log_of_all_examples = np.zeros(num_status)

        # 对所有样本分别进行健康评估，最后通过投票，选择样本数量最多的隶属状态为样本总体的隶属状态
        for i in range(num_examples):
            if not multiple_sensors:
                test_data_selected = test_data[features_list].iloc[i, :].to_list()
                test_data_selected = np.array(test_data_selected).T.astype(np.float64)
            else:
                # test_data_selected = get_multiple_sensors_example(test_data, sensor1_key_list, sensor2_key_list, sensor3_key_list)
                # test_data_selected = np.array(test_data_selected).T.astype(np.float64)
                test_data_selected = test_data[i][:].reshape(-1, 1)
            if algorithm == 'FAHP':
                # 层次分析模糊综合评估
                U = data['U']
                level_matrix = getLevel(func_dict, U, test_data_selected, num_second_level, status_nums)
            elif algorithm == 'AHP':
                # 层次逻辑回归分析法
                test_data_selected = test_data_selected.reshape(1, -1)
                level_matrix = model_pred(func_dict, test_data_selected)
            elif algorithm == 'BHM':
                # 层次朴素贝叶斯评估
                test_data_selected = test_data_selected.reshape(1, -1)
                level_matrix = gnb_pred(func_dict, test_data_selected)
            else:
                # 增值服务健康评估算法
                if extra_algorithm_filepath:
                    # print(f'extra_algorithm_filepath: {extra_algorithm_filepath}')
                    extra_input_filepath = extra_algorithm_user_dir + '/input_data.npy'  # 向专有健康评估算法脚本传递数据的文件路径
                    extra_output_filepath = extra_algorithm_user_dir + '/level_matrix.npy'
                    np.save(extra_input_filepath, test_data_selected)
                    # 以脚本形式运行专有健康评估算法
                    result = subprocess.run(shell=True, capture_output=True,
                                            args=f"cd {extra_algorithm_dir} & python ./{extra_algorithm_name}.py --input-filepath "
                                            f"./{username}/input_data.npy --model-filepath ./{extra_algorithm_name}.pkl --save-filepath ./{username}/level_matrix.npy")
                    error = result.stderr.decode('utf-8', errors='replace')
                    if not error:
                        level_matrix = np.load(extra_output_filepath)
                    else:
                        print(f'Error: {error}')
                        print(f'result: {result}')
                        return {'weights_bar': None, 'result_vector': '专有健康评估算法运行出错'}
                else:
                    return {'weights_bar': None, 'result_vector': '专有算法文件路径不存在'}

            save_path_of_example = save_path + '/example_' + str(i)
            if not os.path.exists(save_path_of_example):
                os.makedirs(save_path_of_example)
            if not multiple_sensors:
                weights_bar = weights_Barplot(W_array, save_path_of_example, time_key_list, fre_key_list, primary_key_list)
            else:
                weights_bar = weights_Barplot_multiple_sensor(W_array, save_path_of_example, sensor1_key_list,
                                                              sensor2_key_list,
                                                              sensor3_key_list, primary_key_list)
            second_metric = [len(vector) for vector in W_array]
            sub_matrices = split_rows(second_metric, level_matrix)
            B = []
            for i in range(len(sub_matrices)):
                B.append(np.dot(W_array[i], sub_matrices[i]))
            result = np.dot(W, np.array(B))
            suggestion = save_path_of_example + "/suggestion.txt"
            fw = open(suggestion, 'w', encoding='gbk')
            fw.write(suggestion_dict[status_names[np.argmax(result)]])

            result_log_of_all_examples[np.argmax(result)] += 1

            plot_y = list(result)
            plot_x = [m for m in status_names]
            plt.figure(figsize=(20, 10))
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams.update({'font.size': 20})
            plt.title("评估结果(状态隶属度)")
            plt.grid(ls=" ", alpha=0.5)
            bars = plt.bar(plot_x, plot_y)
            for bar in bars:
                plt.setp(bar, color=plt.get_cmap('cividis')(bar.get_height() / max(plot_y)))
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center', fontsize=20)
            result_vector = save_path_of_example + '/result.npy'
            result_bar = save_path_of_example + '/barPlot.png'
            np.save(result_vector, result)
            plt.savefig(result_bar)
            plt.close()
            if not multiple_sensors:
                tree = plot_tree(primary_key_list, time_key_list, fre_key_list, save_path_of_example)
            else:
                tree = plot_tree_multiple_sensor(primary_key_list, sensor1_key_list, sensor2_key_list,
                                                 sensor3_key_list, save_path_of_example)
            # 将单个样本的评估结果保存到列表中
            weights_bar_of_all_examples.append(weights_bar)
            result_vector_of_all_examples.append(result_vector)
            result_bar_of_all_examples.append(result_bar)
            suggestion_of_all_examples.append(suggestion)
            tree_of_all_examples.append(tree)
        # 最终根据所有样本的状态隶属评估结果，选择数量最多的隶属状态为总体的隶属状态
        final_result_suggestion = suggestion_dict[status_names[np.argmax(result_log_of_all_examples)]]
        status_of_all_examples = {}
        # 将所有样本的状态隶属评估结果保存到字典中
        for i, status in enumerate(status_names):
            status_of_all_examples[status] = result_log_of_all_examples[i]

        print(f'The final result: {final_result_suggestion}')
        print(f'The final result status of all examples: {status_of_all_examples}')

        return {'weights_bar': weights_bar_of_all_examples, 'result_vector': result_vector_of_all_examples,
                'result_bar': result_bar_of_all_examples, 'suggestion': suggestion_of_all_examples,
                'tree': tree_of_all_examples, 'final_result_suggestion': final_result_suggestion,
                'status_of_all_examples': status_of_all_examples}
    except Exception as e:
        return {'weights_bar': None, 'result_vector': str(e)}


# 多传感器数据的健康评估
# def model_eval_multiple_sensor(data_all, model_path, save_path, algorithm, extra_algorithm_filepath=None, username=None):
#     with open(model_path, 'rb') as file:
#         data = pickle.load(file)
#     try:
#         func_dict = data['function_dict']
#         W = data['Weight_1']
#         W_array = data['Weight_2']
#
#         num_second_level = data['num_second_level']
#         status_nums = data['status_nums']
#         sensor1_key_list = data['sensor1_key_list']
#         sensor2_key_list = data['sensor2_key_list']
#         sensor3_key_list = data['sensor3_key_list']
#         status_names = data['status_names']
#         suggestion_dict = data['suggestion_dict']
#         primary_key_list = data['primary_key_list']
#         examples = get_multiple_sensors_example(data_all, sensor1_key_list, sensor2_key_list, sensor3_key_list)
#
#         print(f"healthEvaluation test data shape: {examples.shape}")
#
#         num_examples = examples.shape[0]
#         weights_bar_of_all_examples = []
#         result_vector_of_all_examples = []
#         result_bar_of_all_examples = []
#         suggestion_of_all_examples = []
#         tree_of_all_examples = []
#
#         num_status = len(status_names)
#         result_log_of_all_examples = np.zeros(num_status)
#
#         for i in range(num_examples):
#             test_data = examples[i][:].reshape(-1, 1)
#
#             if algorithm == 'FAHP':
#                 # 层次分析模糊综合评估
#                 U = data['U']
#                 level_matrix = getLevel(func_dict, U, test_data, num_second_level, status_nums)
#             elif algorithm == 'AHP':
#                 # 层次逻辑回归评估
#                 test_data = test_data.transpose()
#                 level_matrix = model_pred(func_dict, test_data)
#             elif algorithm == 'BHP':
#                 # 层次朴素贝叶斯评估
#                 level_matrix = gnb_pred(func_dict, test_data.T)
#             else:
#                 return
#
#
#             save_path_of_example = save_path + '/example_' + str(i)
#             if not os.path.exists(save_path_of_example):
#                 os.makedirs(save_path_of_example)
#
#             weights_bar = weights_Barplot_multiple_sensor(W_array, save_path_of_example, sensor1_key_list, sensor2_key_list,
#                                                           sensor3_key_list, primary_key_list)
#             second_metric = [len(vector) for vector in W_array]
#             sub_matrices = split_rows(second_metric, level_matrix)
#             B = []
#             for index in range(len(sub_matrices)):
#                 B.append(np.dot(W_array[index], sub_matrices[index]))
#             result = np.dot(W, np.array(B))
#             result = normalize(result.reshape(1, -1), norm='l1').squeeze()
#             suggestion = save_path_of_example + "/suggestion.txt"
#             fw = open(suggestion, 'w', encoding='gbk')
#             fw.write(suggestion_dict[status_names[np.argmax(result)]])
#
#             result_log_of_all_examples[np.argmax(result)] += 1
#
#             plot_y = list(result)
#             plot_x = [m for m in status_names]
#             plt.figure(figsize=(20, 10))
#             plt.rcParams['font.sans-serif'] = ['SimHei']
#             plt.rcParams.update({'font.size': 20})
#             plt.title("评估结果(状态隶属度)")
#             plt.grid(ls=" ", alpha=0.5)
#             bars = plt.bar(plot_x, plot_y)
#             for bar in bars:
#                 plt.setp(bar, color=plt.get_cmap('cividis')(bar.get_height() / max(plot_y)))
#                 yval = bar.get_height()
#                 plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center', fontsize=20)
#             # plt.savefig(save_path + '/barPlot.png')
#             tree = plot_tree_multiple_sensor(primary_key_list, sensor1_key_list, sensor2_key_list,
#                                              sensor3_key_list, save_path_of_example)
#             result_vector = save_path_of_example + '/result.npy'
#             result_bar = save_path_of_example + '/barPlot.png'
#             np.save(result_vector, result)
#             plt.savefig(result_bar)
#             plt.close()
#             # 将单个样本的评估结果保存到列表中
#             weights_bar_of_all_examples.append(weights_bar)
#             result_vector_of_all_examples.append(result_vector)
#             result_bar_of_all_examples.append(result_bar)
#             suggestion_of_all_examples.append(suggestion)
#             tree_of_all_examples.append(tree)
#         # 最终根据所有样本的状态隶属评估结果，选择数量最多的隶属状态为总体的隶属状态
#         final_result_suggestion = suggestion_dict[status_names[np.argmax(result_log_of_all_examples)]]
#         status_of_all_examples = {}
#         # 将所有样本的状态隶属评估结果保存到字典中
#         for i, status in enumerate(status_names):
#             status_of_all_examples[status] = result_log_of_all_examples[i]
#         return {'weights_bar': weights_bar_of_all_examples, 'result_vector': result_vector_of_all_examples,
#                 'result_bar': result_bar_of_all_examples, 'final_result_suggestion': final_result_suggestion,
#                 'suggestion': suggestion_of_all_examples, 'tree': tree_of_all_examples,
#                 'status_of_all_examples': status_of_all_examples}
#     except Exception as e:
#         return {'weights_bar': None, 'result_vector': str(e)}
