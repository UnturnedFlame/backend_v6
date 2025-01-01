import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

# 加载数据
file_path = 'app1/module_management/algorithms/functions/datas/vibration_features_with_labels.csv'  # 请确保文件路径正确
file_path_2 = 'app1/module_management/algorithms/functions/datas/multi_sensor_features.csv'
save_path = 'app1/module_management/algorithms/functions/feature_selection_results'


# 绘制相关系数矩阵热力图
def correlation_matrix_plot(features, figure_save_path, multiple_sensor=False):
    """
    绘制相关系数矩阵热力图
    :param multiple_sensor: 是否为多传感器数据
    :param figure_save_path: 图像的保存路径
    :param features: 相关系数矩阵热力图涉及到的特征名
    :return: 无返回值
    """
    if not multiple_sensor:
        df = pd.read_csv(file_path).loc[:, features]
    else:
        df = pd.read_csv(file_path_2).loc[:, features]
    corr_matrix = df.corr()
    matplotlib.use('Agg')
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题

    # 绘制相关系数图
    if not multiple_sensor:
        plt.figure(figsize=(10, 8))
    else:
        plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 7})
    plt.title('相关系数矩阵热力图')

    plt.savefig(figure_save_path)


# 使用树模型的特征选择
def feature_imp(multiple_sensor=False, rule=1, threshold=0, user_dir=None):
    """
    模型相关的特征选择
    :param user_dir: 结果图像的保存目录
    :param threshold: 特征选择依据规则的设定阈值
    :param rule: 特征选择的依据规则
    :param multiple_sensor: 是否为多传感器数据
    :return: 特征选择结果图像的存放路径
    """
    if not multiple_sensor:
        data = pd.read_csv(file_path)
    else:
        data = pd.read_csv(file_path_2)
    # 阈值范围调整
    if rule == 1:
        threshold /= 10.0
    print(f'feature_imp threshold: {threshold}')
    all_columns = data.columns
    empty_columns = [col for col in all_columns if ('谱峭度的偏度' in col or '谱峭度的标准差' in col)]

    # 删除全空的列
    data_cleaned = data.drop(columns=empty_columns)
    # 分离特征和标签
    X_cleaned = data_cleaned.drop(columns=['label'])
    y_cleaned = data_cleaned['label']

    # 用列的均值填充缺失值
    imputer = SimpleImputer(strategy='mean')
    X_imputed_cleaned = imputer.fit_transform(X_cleaned)

    # 训练随机森林分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_imputed_cleaned, y_cleaned)

    # 获取特征重要性
    importances_cleaned = clf.feature_importances_
    indices_cleaned = np.argsort(importances_cleaned)[::-1]

    num_features_selected = 0
    features_selected = {}

    # 根据规则选择特征
    if rule == 1:
        # 规则一，选择特征的重要性大于阈值threshold的特征
        for i in range(len(indices_cleaned)):
            if importances_cleaned[indices_cleaned[i]] > threshold:
                features_selected[X_cleaned.columns[indices_cleaned[i]]] = importances_cleaned[indices_cleaned[i]]
                num_features_selected += 1
    else:
        # 规则二，所选择特征的重要性的总和占比超过阈
        # 值threshold（所有特征重要性的总和为1），优先选择重要性高的特征
        sum_importance = 0
        importance_summed = np.sum(importances_cleaned).item()
        print(f'重要性的总和{importance_summed}')
        for i in range(len(indices_cleaned)):
            sum_importance += importances_cleaned[indices_cleaned[i]]
            if sum_importance / importance_summed <= threshold:
                features_selected[X_cleaned.columns[indices_cleaned[i]]] = importances_cleaned[indices_cleaned[i]]
                num_features_selected += 1

    # features_selected = {X_cleaned.columns[indices_cleaned[i]]: importances_cleaned[indices_cleaned[i]]
    #                      for i in range(len(indices_cleaned)) if importances_cleaned[indices_cleaned[i]] > threshold}
    matplotlib.use('Agg')
    # 可视化特征重要性
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题

    plt.figure(figsize=(20, 10))

    plt.title("树模型特征选择", fontsize=20)
    if not multiple_sensor:
        bars = plt.bar(range(len(importances_cleaned)), importances_cleaned[indices_cleaned], align="center")
        plt.xticks(range(len(importances_cleaned)), X_cleaned.columns[indices_cleaned], rotation=90, fontsize=12)
    else:
        if len(X_cleaned.columns) > 30:
            end_of_view = 30
        else:
            end_of_view = len(X_cleaned.columns)
        bars = plt.bar(range(end_of_view), importances_cleaned[indices_cleaned][0:end_of_view], align="center")
        plt.xticks(range(end_of_view), X_cleaned.columns[indices_cleaned][0:end_of_view], rotation=90, fontsize=12)

    # 使用不同颜色区分被选择的特征
    for i in range(num_features_selected):
        bars[i].set_color('r')
    features_selected = {X_cleaned.columns[indices_cleaned[i]]: importances_cleaned[indices_cleaned[i]]
                         for i in range(num_features_selected)}
    # plt.xticks(range(len(importances_cleaned)), X_cleaned.columns[indices_cleaned], rotation=90, fontsize=20)
    # plt.xlim([-1, len(importances_cleaned)])
    # plt.tight_layout()
    plt.ylabel('特征重要性', fontsize=20)
    # 结果图像保存
    figure_save_path = save_path + '/' + 'feature_imp' + '/' + user_dir
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    corr_matrix_plot_path = figure_save_path + '/' + 'corr_matrix_heatmap.png'
    selection_figure_save_path = figure_save_path + '/' + 'selection_result.png'

    # 特征选择结果图
    plt.savefig(selection_figure_save_path)
    # 相关系数矩阵热力图
    correlation_matrix_plot(list(features_selected.keys()), corr_matrix_plot_path, multiple_sensor=multiple_sensor)
    plt.close()

    return selection_figure_save_path, list(features_selected.keys()), corr_matrix_plot_path


# 绘制特征选择结果图像
def plot_importance(importance_dict, title, multiple_sensor=False, rule=1, threshold=0, user_dir=None):
    """
    根据计算得到的所有特征的重要性以及对应的规则及阈值进行特征选择，然后绘制特征选择结果图像
    :param user_dir: 结果的保存目录
    :param threshold: 选择特征的规则对应的阈值
    :param rule: 选择特征的规则
    :param importance_dict: 重要性字典
    :param title: 使用的特征选择方法
    :param multiple_sensor: 是否为多传感器数据
    :return: 绘制图像的存放路径
    """
    sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
    features = [item[0] for item in sorted_importance]
    scores = [item[1] for item in sorted_importance]

    features_selected = []  # 选择的特征名称
    num_features_selected = 0  # 选择的特征数量
    sum_importance = 0     # 所选特征的重要性的总和
    importance_summed = sum(scores)  # 所有的特征的重要性的总和

    if rule == 1:
        # 应用规则一
        for feature, score in zip(features, scores):
            if score > threshold:
                features_selected.append(feature)
                num_features_selected += 1
    else:
        # 应用规则二
        for feature, score in zip(features, scores):
            sum_importance += score
            if sum_importance/importance_summed <= threshold:
                features_selected.append(feature)
                num_features_selected += 1

    matplotlib.use('Agg')
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题

    if not multiple_sensor:
        plt.figure(figsize=(20, 10))
        plt.title(title, fontsize=20)
        bars = plt.bar(range(len(scores)), scores, align="center")
        plt.xticks(range(len(scores)), features, rotation=90, fontsize=12)
    else:
        plt.figure(figsize=(20, 10))
        plt.title(title, fontsize=20)
        if num_features_selected + 5 >= len(scores):
            boundary = num_features_selected
        else:
            boundary = num_features_selected + 5

        bars = plt.bar(range(boundary), scores[0:boundary], align="center")
        plt.xticks(range(boundary), features[0:boundary], rotation=90, fontsize=12)

    # 使用不同颜色区分被选择的特征
    for i in range(num_features_selected):
        bars[i].set_color('r')

    # plt.tight_layout()
    if title == "互信息重要性特征选择":
        filename = "mutual_information_importance.png"
        figure_save_path = save_path + "/mutual_information_importance/" + user_dir
        plt.ylabel('互信息重要性', fontsize=20)
    else:
        filename = "correlation_coefficient_importance.png"
        figure_save_path = save_path + "/correlation_coefficient_importance/" + user_dir
        plt.ylabel('相关系数重要性', fontsize=20)
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    figure_save_path += '/' + filename
    plt.savefig(figure_save_path)
    plt.close()

    return figure_save_path, features_selected


def mutual_information_importance(multiple_sensor=False, rule=1, threshold=0, user_dir=None):
    """
    互信息重要性的特征选择
    :param user_dir: 结果的保存目录
    :param threshold: 特征选择依据的规则的阈值
    :param rule: 特征选择依据的规则
    :param multiple_sensor: 是否为多传感器数据
    :return: 特征选择的结果图像的存放路径
    """
    if not multiple_sensor:
        data = pd.read_csv(file_path)
    else:
        data = pd.read_csv(file_path_2)
    all_columns = data.columns
    empty_columns = [col for col in all_columns if ('谱峭度的偏度' in col or '谱峭度的标准差' in col)]
    # 删除全空的列
    # data_cleaned = data.drop(columns=['谱峭度的偏度', '谱峭度的标准差'])
    data_cleaned = data.drop(columns=empty_columns)
    # 分离特征和标签
    X_cleaned = data_cleaned.drop(columns=['label'])
    y_cleaned = data_cleaned['label']

    # 互信息法
    mi = mutual_info_classif(X_cleaned, y_cleaned, discrete_features='auto')
    mi_importance = {X_cleaned.columns[i]: mi[i] for i in range(len(mi))}

    # 可视化互信息重要性
    figure_path, features = plot_importance(mi_importance, "互信息重要性特征选择", multiple_sensor=multiple_sensor, rule=rule, threshold=threshold, user_dir=user_dir)
    corr_matrix_heatmap_path = save_path + '/' + 'mutual_information_importance' + '/' + user_dir
    if not os.path.exists(corr_matrix_heatmap_path):
        os.makedirs(corr_matrix_heatmap_path)
    corr_matrix_heatmap_path += '/' + 'corr_matrix_heatmap.png'
    correlation_matrix_plot(features, corr_matrix_heatmap_path, multiple_sensor=multiple_sensor)
    return figure_path, features, corr_matrix_heatmap_path


def correlation_coefficient_importance(multiple_sensor=False, rule=1, threshold=0, user_dir=None):
    """
    相关系数重要性的特征选择
    :param user_dir: 结果的保存目录
    :param threshold: 特征选择依据的规则的阈值
    :param rule: 特征选择依据的规则
    :param multiple_sensor: 是否为多传感器数据
    :return: 特征选择的结果图像的存放路径
    """
    if not multiple_sensor:
        data = pd.read_csv(file_path)
    else:
        data = pd.read_csv(file_path_2)

    all_columns = data.columns
    empty_columns = [col for col in all_columns if ('谱峭度的偏度' in col or '谱峭度的标准差' in col)]
    # 删除全空的列
    # data_cleaned = data.drop(columns=['谱峭度的偏度', '谱峭度的标准差'])
    data_cleaned = data.drop(columns=empty_columns)
    # 分离特征和标签
    X_cleaned = data_cleaned.drop(columns=['label'])
    y_cleaned = data_cleaned['label']

    # 相关系数法
    correlations = [pearsonr(X_cleaned[col], y_cleaned)[0] for col in X_cleaned.columns]
    cor_importance = {X_cleaned.columns[i]: abs(correlations[i]) for i in range(len(correlations))}

    # 可视化相关系数重要性
    figure_path, features = plot_importance(cor_importance, "相关系数重要性特征选择", multiple_sensor=multiple_sensor, rule=rule, threshold=threshold, user_dir=user_dir)
    corr_matrix_heatmap_path = save_path + '/' + 'correlation_coefficient_importance' + '/' + user_dir
    if not os.path.exists(corr_matrix_heatmap_path):
        os.makedirs(corr_matrix_heatmap_path)
    corr_matrix_heatmap_path += '/' + 'corr_matrix_heatmap.png'
    correlation_matrix_plot(features, corr_matrix_heatmap_path, multiple_sensor=multiple_sensor)

    return figure_path, features, corr_matrix_heatmap_path
