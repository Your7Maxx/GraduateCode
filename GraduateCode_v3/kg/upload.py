from py2neo import Graph, Node, Relationship
from datetime import datetime
import requests

# 连接到 Neo4j 数据库
graph = Graph("bolt://localhost:7687", user="neo4j", password="123456")

# 读取数据文件并解析数据
data_file = "command_data.test2"  # 请根据实际文件路径修改

def get_class_probability(command):
    url = 'http://localhost:5000/predict'  # 请根据实际情况修改 URL
    headers = {'Content-Type': 'application/json'}
    data = {'new_command': command}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        class_probabilities = response_data.get("Class Probabilities", [[]])[0]
        if class_probabilities:
            return class_probabilities[1]  # 取第二个值
    return None

def get_anomaly_score(timestamp, ppcomm, pcomm):
    url = "http://localhost:5001/anomaly_score"  # 替换成你的Flask应用的URL

    data = {
        "timestamp": timestamp,
        "ppcomm": ppcomm,
        "pcomm": pcomm
    }

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            return result.get('anomaly_score')
        else:
            print(f"Request failed with status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None


with open(data_file, 'r') as file:
    query1 = """
        match (n) detach delete (n)
        """
    result1 = graph.run(query1).data()

    query2 = """
        call gds.graph.drop("myGraph")
        """
#    result2 = graph.run(query2).data()

    for line in file:
        try:

            data = line.strip().split('"')
            #print(data)
            uid, time, child_comm, child_pid, parent_comm, parent_pid, return_code, cmdline, option = data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15], data[17]
            #print(timestamp, child_comm, child_pid, parent_comm, parent_pid, return_code, cmdline)
            date_object = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            timestamp = date_object.timestamp()
            label = int(line.strip()[-1])
            id = str(time) + '-' + str(child_pid)
            # 创建父进程和子进程节点
            parent_process = Node("Process", id=id, pid=parent_pid, name=parent_comm, comm=parent_comm)
            child_process = Node("Process", id=id,pid=child_pid, name=child_comm, comm=child_comm)

            # 使用 Relationship() 创建关系
            rel = Relationship(parent_process, "create_process", child_process)
            rel["uid"] = uid
            rel["time"] = timestamp
            rel["cmdline"] = cmdline
            rel["return"] = return_code
            rel["option"] = option
            rel["label"] = label
            #rel["evil_weight"] = get_class_probability(cmdline)
            #rel["anomaly_score"] = get_anomaly_score(timestamp,parent_comm,child_comm)

            # 使用 graph.merge 来合并节点和关系，确保它们在数据库中是唯一的
            graph.merge(parent_process, "Process", "pid")
            graph.merge(child_process, "Process", "pid")
            graph.merge(rel)
        except Exception as e:
            pass

    query3 = """
            CALL gds.graph.project(
            'myGraph',
            'Process',
            {
                create_process: {
                    orientation: 'UNDIRECTED'
                }
            },
            {
                relationshipProperties: 'time'
            })
        """
    result3 = graph.run(query3).data()

print("Data imported into Neo4j database.")

