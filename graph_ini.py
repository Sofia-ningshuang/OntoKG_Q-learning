import json
import networkx as nx
from networkx.readwrite import json_graph

# 为图的边设置初始权重
def get_edge_weight(edge_data):
    """
    Determine edge weight based on the edge label or edge_type.
    Rules:
    - "IsA" or "inheritance" -> 5
    - "CorrelatesWith" -> 3
    - "DirectlyImpacts" -> 5
    - "DirectlyCauses" -> 7
    - Default -> 1
    """
    # Check edge_type first 检查边类型
    edge_type = edge_data.get('edge_type', '')
    label = edge_data.get('label', '')
    
    # Combine both for checking 根据边类型与标签进行初步赋权
    edge_info = f"{edge_type} {label}".lower()
    
    if 'isa' in edge_info:
        return 5
    if 'inheritance' in edge_info:
        return 5
    elif 'correlateswith' in edge_info:
        return 3
    elif 'directlyimpacts' in edge_info:
        return 5
    elif 'directlycauses' in edge_info:
        return 7
    else:
        return 1  # Default weight 默认初始权重，可根据实际情况修改，可外部读取，也可结合概率分布

# 加载图进行重构与赋权，为Q-learning做准备
def load_graph_for_qlearning(json_file='ontology_graph.json'):
    """
    Load the ontology graph from JSON and prepare it for Q-learning.
    Adds weights to edges based on relationship types.
    """
    # Load the graph from JSON 从JSON加载图
    with open(json_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # Reconstruct the NetworkX graph 重构图以便进行计算
    G = json_graph.node_link_graph(graph_data)
    
    # Add weights to all edges 边赋权
    for u, v, data in G.edges(data=True):
        weight = get_edge_weight(data)
        # If source node is the generic 'Instance' class, set weight to 0 as requested 如果是泛意义"实例"类，将权重设置为0
        u_label = G.nodes[u].get('label')
        if u_label == 'Instance':
            weight = 0
        G[u][v]['weight'] = weight
    
    print(f"Loaded graph for Q-learning:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Print edge weight distribution 输出边权重分布
    weight_counts = {}
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1)
        weight_counts[w] = weight_counts.get(w, 0) + 1
    
    print(f"\nEdge weight distribution:")
    for weight in sorted(weight_counts.keys()):
        print(f"  Weight {weight}: {weight_counts[weight]} edges")
    
    return G

# 输出节点信息
def print_graph_info(G):
    """Print detailed information about the graph."""
    print(f"\n{'='*60}")
    print(f"Graph Information")
    print(f"{'='*60}")
    
    # Node types 整理节点类型
    node_types = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"\nNode types:")
    for ntype, count in node_types.items():
        print(f"  {ntype}: {count}")
    
    # Edge types 整理边类型
    edge_types = {}
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print(f"\nEdge types:")
    for etype, count in edge_types.items():
        print(f"  {etype}: {count}")
    
    # Sample nodes 示例节点
    print(f"\nSample nodes (first 5):")
    for i, (node, data) in enumerate(G.nodes(data=True)):
        if i >= 5:
            break
        label = data.get('label', node)
        node_type = data.get('node_type', 'unknown')
        print(f"  {label} (type: {node_type})")
    
    # Sample edges with weights 示例边
    print(f"\nSample edges with weights (first 5):")
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        if i >= 5:
            break
        u_label = G.nodes[u].get('label', u)
        v_label = G.nodes[v].get('label', v)
        weight = data.get('weight', 1)
        edge_label = data.get('label', data.get('edge_type', ''))
        print(f"  {u_label} -> {v_label} (weight: {weight}, label: {edge_label})")

if __name__ == "__main__":
    # Load the graph 加载图
    G = load_graph_for_qlearning('ontology_graph.json')
    
    # Print detailed information 输出细节信息
    print_graph_info(G)
    
    print(f"\n{'='*60}")
    print("Graph is ready for Q-learning!")
    print(f"{'='*60}")
