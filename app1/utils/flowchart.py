from graphviz import Digraph

def create_model_diagram(modules):
    dot = Digraph(comment='Model Flow', format='png', encoding='utf-8')

    dot.attr(fontname='MicrosoftYaHei', encoding='utf-8',rankdir='LR')  # 改为系统中的中文字体
    dot.node_attr.update(fontname='MicrosoftYaHei')  # 设置节点字体
    dot.edge_attr.update(fontname='MicrosoftYaHei')  # 设置边字体
    # dot.attr()  # 从左到右排列

    # 添加节点
    for module in modules:
        dot.node(module, module, shape='box', style='filled', color='lightblue')

    # 添加边
    for i in range(len(modules) - 1):
        dot.edge(modules[i], modules[i + 1])

    # 保存并显示
    dot.render('model_flow', view=True)
    return dot


# 使用示例
# modules = ['特征提取', '特征选择', '故障诊断', '健康评估']
# diagram = create_model_diagram(modules)


def create_detailed_flowchart(modules, inputs=None, outputs=None, params=None):
    """
    创建带详细信息的流程图

    参数:
    modules: 模块顺序列表
    inputs: 各模块输入描述 {模块名: 输入描述}
    outputs: 各模块输出描述 {模块名: 输出描述}
    params: 各模块参数描述 {模块名: 参数描述}
    """
    dot = Digraph(comment='Detailed Model Flow', format='png', encoding='utf-8')
    dot.node_attr.update(fontname='MicrosoftYaHei')  # 设置节点字体
    dot.edge_attr.update(fontname='MicrosoftYaHei')  # 设置边字体
    dot.attr(fontname='MicrosoftYaHei', encoding='utf-8', rankdir='LR')  # 从左到右排列

    # 添加输入节点
    if inputs:
        dot.node('input', '输入数据\n' + inputs.get('global', ''),
                 shape='parallelogram', style='filled', fillcolor='lightgreen')
        dot.edge('input', modules[0])

    # 添加处理模块
    for module in modules:
        # 构建节点标签内容
        label = f'<<B>{module}</B>'

        if params and module in params:
            label += f'<BR/><FONT POINT-SIZE="10">参数: {params[module]}</FONT>'

        label += '>'

        dot.node(module, label, shape='box', style='filled',
                 fillcolor='lightblue', fontname='Microsoft YaHei')

    # 添加模块间连接
    for i in range(len(modules) - 1):
        if outputs and modules[i] in outputs:
            dot.edge(modules[i], modules[i + 1],
                     label=f' {outputs.get(modules[i], "")} ', fontsize='10')
        else:
            dot.edge(modules[i], modules[i + 1])

    # 添加输出节点
    if outputs:
        last_module = modules[-1]
        dot.node('output', '输出结果\n' + outputs.get(last_module, ''),
                 shape='parallelogram', style='filled', fillcolor='lightpink')
        dot.edge(last_module, 'output')

    dot.render('detailed_model_flow', view=True)
    return dot


# 使用示例


if __name__ == '__main__':
    modules = ['特征提取', '特征选择', '故障诊断', '健康评估']

    inputs = {
        'global': '传感器原始数据\n采样频率: 10kHz',
        '特征提取': '时域信号'
    }

    outputs = {
        '特征提取': '100维特征向量',
        '特征选择': '20维关键特征',
        '故障诊断': '故障类型及概率',
        '健康评估': '健康评分(0-1)'
    }

    params = {
        '特征提取': '小波变换层数=5',
        '特征选择': 'SelectKBest(k=20)',
        '故障诊断': 'SVM(C=1.0, kernel=rbf)',
        '健康评估': '阈值=0.7, 随机值2222222'
    }

    create_detailed_flowchart(modules, inputs, outputs, params)