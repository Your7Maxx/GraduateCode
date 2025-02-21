from py2neo import Graph
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def run_graph_and_cluster():
    # 连接到 Neo4j 图数据库
    graph = Graph("bolt://localhost:7687", user="neo4j", password="123456")

    # 查询所有连通子图的 componentId 和节点的关系
    query = """
        CALL gds.wcc.stream('myGraph')
        YIELD nodeId, componentId
        MATCH (n)-[r]->(m)
        WHERE id(n) = nodeId
        RETURN componentId, n, r, m
        ORDER BY componentId ASC
    """  
    result = graph.run(query).data()

    # 按 componentId 分组
    component_results = {}
    for record in result:
        component_id = record['componentId']
        if component_id not in component_results:
            component_results[component_id] = []
        component_results[component_id].append({
            'node': record['n'],
            'relationship': record['r'],
            'target_node': record['m']
        })
    
    # 存储所有聚类的结果（包括没有参与聚类的）
    all_clusters = []

    # 遍历每个连通子图并提取数据
    for component_id, nodes_and_relationships in component_results.items():
        data = []  # 存储数据
        original_data = []  # 存储原始数据
        
        for item in nodes_and_relationships:
            ppnode = item['node']
            relationship = item['relationship']
            pnode = item['target_node']
            
            # 从关系中提取信息
            uid = relationship['uid']
            timestamp = relationship['time']
            pcomm = pnode['name']
            ppcomm = ppnode['name']
            ppid = ppnode['pid']
            option = relationship['option']
            
            # 将数据添加到数据列表
            data.append([uid, timestamp])
            original_data.append([uid, timestamp, pcomm, ppcomm, ppid, option])  # 保留原始数据
        
        # 如果该子图的样本数小于 2，则跳过聚类，单独处理
        n_samples = len(data)
        if n_samples < 3:
            all_clusters.append([original_data])  # 将数据放入一个独立的聚类
            continue  # 跳过聚类

        # 将数据转换为 NumPy 数组进行聚类
        data_array = np.array(data)

        # 提取聚类特征：这里使用 uid 和 timestamp 作为聚类特征
        clustering_data = data_array[:, [0, 1]].astype(float)  # 选择 uid 和 timestamp 列

        # 进行 KMeans 聚类
        n_clusters = 3  # 设定默认簇数
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(clustering_data)

        # 存储聚类结果
        clusters = [[] for _ in range(n_clusters)]
        
        for original_item, cluster_label in zip(original_data, cluster_labels):
            clusters[cluster_label].append(original_item)  # 将数据添加到对应的聚类中
        
        # 将聚类结果添加到 all_clusters 中
        all_clusters.append(clusters)

    return all_clusters

def run_graph_and_cluster2():
    # 连接到 Neo4j 图数据库
    graph = Graph("bolt://localhost:7687", user="neo4j", password="123456")

    # 查询所有连通子图的 componentId 和节点的关系
    query = """
        CALL gds.wcc.stream('myGraph')
        YIELD nodeId, componentId
        MATCH (n)-[r]->(m)
        WHERE id(n) = nodeId
        RETURN componentId, n, r, m
        ORDER BY componentId ASC
    """
    result = graph.run(query).data()

    # 按 componentId 分组
    component_results = {}
    for record in result:
        component_id = record['componentId']
        if component_id not in component_results:
            component_results[component_id] = []
        component_results[component_id].append({
            'node': record['n'],
            'relationship': record['r'],
            'target_node': record['m']
        })

    # 存储所有聚类的结果（包括没有参与聚类的）
    all_clusters = []

    # 遍历每个连通子图并提取数据
    for component_id, nodes_and_relationships in component_results.items():
        data = []  # 存储数据
        original_data = []  # 存储原始数据

        for item in nodes_and_relationships:
            ppnode = item['node']
            relationship = item['relationship']
            pnode = item['target_node']

            # 从关系中提取信息
            uid = relationship['uid']
            timestamp = relationship['time']
            pcomm = pnode['name']
            ppcomm = ppnode['name']
            ppid = ppnode['pid']
            option = relationship['option']

            # 将数据添加到数据列表
            data.append([uid, timestamp])
            original_data.append([uid, timestamp, pcomm, ppcomm, ppid, option])  # 保留原始数据

        # 如果该子图的样本数小于 2，则跳过聚类，单独处理
        n_samples = len(data)
        if n_samples < 3:
            all_clusters.append([original_data])  # 将数据放入一个独立的聚类
            continue  # 跳过聚类

        # 将数据转换为 NumPy 数组进行聚类
        data_array = np.array(data)

        # 提取聚类特征：这里使用 uid 和 timestamp 作为聚类特征
        clustering_data = data_array[:, [0, 1]].astype(float)  # 选择 uid 和 timestamp 列

        # 进行 KMeans 聚类
        n_clusters = 3  # 设定默认簇数
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(clustering_data)

        # 存储聚类结果
        clusters = [[] for _ in range(n_clusters)]

        for original_item, cluster_label in zip(original_data, cluster_labels):
            clusters[cluster_label].append(original_item)  # 将数据添加到对应的聚类中

        # 将聚类结果添加到 all_clusters 中
        all_clusters.append(clusters)

    return all_clusters

def find_similar_entries_in_cluster(entry, all_clusters):
    """
    给定一个条目，查找该条目所在的聚类并返回该聚类中的其他条目。

    :param entry: 条目，格式为 [timestamp, uid, ppcomm, pcomm, ppid, option]
    :param all_clusters: 所有聚类的结果
    :return: 该条目所在聚类的其他条目
    """
    # 遍历所有聚类，寻找给定条目所在的聚类
    for clusters in all_clusters:
        for cluster in clusters:
            # 检查当前条目是否在该聚类中
            if entry in cluster:
                # 返回该聚类中所有条目，排除掉当前条目
                return [item for item in cluster if item != entry]

    # 如果条目未找到，返回空列表
    return []

def run_graph_and_detect_anomalies():
    # 连接到 Neo4j 图数据库
    graph = Graph("bolt://localhost:7687", user="neo4j", password="123456")

    # 查询所有连通子图的 componentId 和节点的关系
    query = """
        CALL gds.wcc.stream('myGraph')
        YIELD nodeId, componentId
        MATCH (n)-[r]->(m)
        WHERE id(n) = nodeId
        RETURN componentId, n, r, m
        ORDER BY componentId ASC
    """  
    result = graph.run(query).data()

    # 按 componentId 分组
    component_results = {}
    for record in result:
        component_id = record['componentId']
        if component_id not in component_results:
            component_results[component_id] = []
        component_results[component_id].append({
            'node': record['n'],
            'relationship': record['r'],
            'target_node': record['m']
        })

    # 存储 FPR 和 contamination 对应的结果
    contamination_values = np.arange(0.01, 0.31, 0.01)
    fpr_values = []

    # 遍历 contamination 参数
    for contamination in contamination_values:
        all_labels = []
        all_predictions = []
        all_clusters = []  # 存储聚类结果

        # 遍历每个连通子图并提取数据
        for component_id in component_results:
            data = []
            labels = []

            for record in component_results[component_id]:
                ppnode = record['node']
                relationship = record['relationship']
                pnode = record['target_node']
                uid, timestamp, pcomm, ppcomm, ppid, option, label = relationship['uid'], relationship['time'], pnode['name'], ppnode['name'], ppnode['pid'], relationship['option'], relationship['label']
                data.append([uid, timestamp, pcomm, ppcomm, ppid, option])
                labels.append(label)
            
            # 将数据转换为 numpy 数组
            data_tmp = data
            data = np.array(data)

            # 对 timestamp 进行标准化处理
            scaler = MinMaxScaler()
            data[:, 1] = scaler.fit_transform(data[:, 1].reshape(-1, 1)).flatten()

            # 将其他特征编码为整数
            le_process = LabelEncoder()
            data[:, 0] = le_process.fit_transform(data[:, 0])
            data[:, 2] = le_process.fit_transform(data[:, 2])
            data[:, 3] = le_process.fit_transform(data[:, 3])
            data[:, 4] = le_process.fit_transform(data[:, 4])
            data[:, 5] = le_process.fit_transform(data[:, 5])

            # 创建孤立森林模型
            clf = IsolationForest(contamination=contamination, random_state=42)
            clf.fit(data)

            # 获取预测结果
            predictions = clf.predict(data)

            # 运行聚类函数
            all_clusters = run_graph_and_cluster()

            # 处理异常预测结果
            for j, pred in enumerate(predictions):
                if pred == -1:
                    # 查找异常条目所在的聚类并输出其他条目
                    similar_entries = find_similar_entries_in_cluster(data_tmp[j], all_clusters)

                    # 标记聚类中的其他条目为异常
                    for entry in similar_entries:
                        idx = data_tmp.index(entry)
                        predictions[idx] = -1  # 将关联条目标记为异常

            # 更新预测结果
            predictions = [1 if pred == -1 else 0 for pred in predictions]
            all_labels.append(labels)
            all_predictions.append(predictions)

        # 将所有社区的 labels 和 predictions 展开成单个列表
        all_labels = [label for sublist in all_labels for label in sublist]
        all_predictions = [prediction for sublist in all_predictions for prediction in sublist]

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # 从混淆矩阵中提取 TP, TN, FP, FN
        TP = conf_matrix[1, 1]  # 真阳性
        TN = conf_matrix[0, 0]  # 真阴性
        FP = conf_matrix[0, 1]  # 假阳性
        FN = conf_matrix[1, 0]  # 假阴性

        # 计算 FPR（假阳性率）
        fpr = FP / (FP + TN)
        fpr_values.append(fpr)

    # 绘制 FPR 曲线图
    plt.plot(contamination_values, fpr_values, marker='o', linestyle='-', color='b')
    plt.title('FPR vs Contamination')
    plt.xlabel('Contamination')
    plt.ylabel('FPR')
    plt.grid(True)
    plt.show()

run_graph_and_detect_anomalies()
