from py2neo import Graph

# 建立与Neo4j数据库的连接
graph = Graph("bolt://localhost:7687", user="neo4j", password="123456")

# 第一个查询，获取所有不重复的communityId
query1 = """
CALL gds.louvain.stream('myGraph')
YIELD nodeId, communityId, intermediateCommunityIds
RETURN DISTINCT communityId
ORDER BY communityId ASC
"""

result1 = graph.run(query1).data()

# 存储所有不重复的communityId
community_ids = [record['communityId'] for record in result1]

# 存储每个community的结果
community_results = []

# 第二个查询，对每个communityId执行
for community_id in community_ids:
    query2 = f"""
    CALL gds.louvain.stream('myGraph')
    YIELD nodeId, communityId, intermediateCommunityIds
    WHERE communityId = {community_id}
    MATCH (n)-[r]->(m)
    WHERE id(n) = nodeId
    RETURN n, r, m
    ORDER BY communityId ASC
    """
    result2 = graph.run(query2).data()
    community_results.append(result2)

# 打印结果
for i, community_id in enumerate(community_ids):
    print("----"*10)
    print(f"Community {community_id}:")
    for record in community_results[i]:
        print("----"*10)
        node = record['n']
        relationship = record['r']
        related_node = record['m']
        print(f"PPCOMM: {node['name']}")
        print(f"Rel: {relationship}")
        print(f"PCOMM: {related_node['name']}")

