import json
from typing import Dict, List

import networkx as nx

from graph_ini import load_graph_for_qlearning
from graph_q_learning import (
    q_learning_to_target,
    enumerate_paths_to_target,
    path_scores,
    label_path,
    find_node_by_label,
)

# 基于证据的奖励更新
def update_rewards_for_evidence(
    G: nx.DiGraph,
    evidence_weights: Dict[str, int],
    edge_types: List[str] = None,
) -> int:
    """
    Boost rewards (weights) for edges involving evidence nodes.

    - evidence_weights: mapping from node label -> new weight (e.g., {"ReinforcementCageDisplacement": 1}) 当此项原因为排除或者证实可能性很低，权重急剧降低
    - edge_types: if provided, only update edges whose edge_type is in this list.
                  By default, only object_property edges are updated (includes IsA, DirectlyImpacts, etc.).

    Returns the count of edges updated.
    """
    if edge_types is None:
        edge_types = ["object_property"]

    updated = 0
    for u, v, data in G.edges(data=True):
        et = data.get("edge_type")
        if et not in edge_types:
            continue
        u_lab = G.nodes[u].get("label")
        v_lab = G.nodes[v].get("label")
        new_w = None
        if u_lab in evidence_weights:
            new_w = evidence_weights[u_lab]
        if v_lab in evidence_weights:
            new_w = max(new_w or 0, evidence_weights[v_lab])
        if new_w is not None:
            old_w = data.get("weight", 0)
            if old_w != new_w:
                G[u][v]["weight"] = new_w
                updated += 1
    return updated

# 基于证据进行Q-learning
def run_with_evidence(
    target_label: str = "PileHiddenDefect",
    evidence_weights: Dict[str, int] = None,
    episodes: int = 3000,
    max_depth: int = 7,
):
    if evidence_weights is None:
        evidence_weights = {"ReinforcementCageDisplacement": 7}

    # Load baseline weighted graph 加载基线加权图
    G = load_graph_for_qlearning("ontology_graph.json")

    # Apply evidence-based reward boosts 启动基于证据的奖励更新
    changed = update_rewards_for_evidence(G, evidence_weights)
    print(f"Boosted rewards on {changed} edges based on evidence: {evidence_weights}")

    # Find target node 搜索目标节点
    target = find_node_by_label(G, target_label)
    if target is None:
        raise ValueError(f"Target '{target_label}' not found in graph labels.")

    # Train Q-values on the modified graph 训练q值
    Q = q_learning_to_target(G, target, episodes=episodes)

    # Enumerate and score paths 枚举并对路径打分
    paths = enumerate_paths_to_target(G, target, max_depth=max_depth)

    scored = []
    for p in paths:
        q_sum, w_sum = path_scores(G, Q, p)
        scored.append({
            "path_nodes": p,
            "path_labels": label_path(G, p),
            "q_sum": q_sum,
            "weight_sum": w_sum,
            "length": len(p) - 1,
        })

    scored.sort(key=lambda x: (x["q_sum"], x["weight_sum"]), reverse=True)

    # Print top 20 updated paths 输出前20条路径
    print("==== Updated Q-learning backtrace results (with evidence) ====")
    for i, s in enumerate(scored[:20], 1):
        print(f"#{i}: q_sum={s['q_sum']:.3f} weight_sum={s['weight_sum']} len={s['length']}")
        print("   " + " -> ".join(s["path_labels"]))

    # Save updated results separately 分开存储更新后结果
    out = {
        "target_label": target_label,
        "episodes": episodes,
        "max_depth": max_depth,
        "evidence_weights": evidence_weights,
        "paths": scored,
    }
    with open(f"q_paths_{target_label}_evidence.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved updated path scores to q_paths_{target_label}_evidence.json (total {len(scored)} paths)")

# 对比结果
def compare_results(
    baseline_file: str = "q_paths_PileHiddenDefect.json",
    updated_file: str = "q_paths_PileHiddenDefect_evidence.json",
    top_k: int = 10,
):
    """Compare baseline vs updated path rankings by path_labels sequence.
    Prints rank changes and Q-score deltas.
    """
    with open(baseline_file, 'r', encoding='utf-8') as f:
        base = json.load(f)
    with open(updated_file, 'r', encoding='utf-8') as f:
        upd = json.load(f)

    base_paths = base.get('paths', [])
    upd_paths = upd.get('paths', [])

    def key(p):
        # Use tuple of labels to identify a path 使用标签元组来标识路径
        return tuple(p.get('path_labels', []))

    base_rank = {key(p): i for i, p in enumerate(base_paths)}
    base_q = {key(p): p.get('q_sum', 0.0) for p in base_paths}
    base_w = {key(p): p.get('weight_sum', 0.0) for p in base_paths}

    upd_rank = {key(p): i for i, p in enumerate(upd_paths)}
    upd_q = {key(p): p.get('q_sum', 0.0) for p in upd_paths}
    upd_w = {key(p): p.get('weight_sum', 0.0) for p in upd_paths}

    common_keys = list(set(base_rank.keys()) & set(upd_rank.keys()))
    only_in_updated = [k for k in upd_rank.keys() if k not in base_rank]
    only_in_baseline = [k for k in base_rank.keys() if k not in upd_rank]

    print("\n==== Comparison: Updated vs Baseline ====")
    print(f"Baseline paths: {len(base_paths)}, Updated paths: {len(upd_paths)}")
    print(f"Common: {len(common_keys)}, Only in Updated: {len(only_in_updated)}, Only in Baseline: {len(only_in_baseline)}")

    # Show top_k by updated rank with deltas 按更新后的排名及其变化量显示前 k 个排名
    rows = []
    for k in common_keys:
        rows.append({
            'labels': k,
            'base_rank': base_rank[k]+1,
            'upd_rank': upd_rank[k]+1,
            'rank_delta': (base_rank[k] - upd_rank[k]),  # positive means improved in updated 正数表示排名上升
            'base_q': base_q[k],
            'upd_q': upd_q[k],
            'q_delta': upd_q[k] - base_q[k],
            'base_w': base_w[k],
            'upd_w': upd_w[k],
            'w_delta': upd_w[k] - base_w[k],
        })
    rows.sort(key=lambda r: r['upd_rank'])

    print(f"\nTop {top_k} by Updated ranking with deltas:")
    for r in rows[:top_k]:
        path_str = " -> ".join(r['labels'])
        print(f"#{r['upd_rank']:>2} (was #{r['base_rank']:>2}, Δrank={r['rank_delta']:+})  ΔQ={r['q_delta']:+.3f}  ΔWeight={r['w_delta']:+.1f}")
        print(f"   {path_str}")

    # Biggest Q increases q值变化最大的情况
    rows_sorted_q = sorted(rows, key=lambda r: r['q_delta'], reverse=True)
    print(f"\nTop {top_k} paths by Q increase:")
    for r in rows_sorted_q[:top_k]:
        path_str = " -> ".join(r['labels'])
        print(f"ΔQ={r['q_delta']:+.3f}  (updated #{r['upd_rank']}, baseline #{r['base_rank']})")
        print(f"   {path_str}")

    # New paths introduced in updated 对更新后的路径进行排序
    if only_in_updated:
        print(f"\nTop {min(top_k, len(only_in_updated))} new paths (only in Updated):")
        # Order by updated rank 根据新的排名更新排序
        only_in_updated.sort(key=lambda k: upd_rank[k])
        for k in only_in_updated[:top_k]:
            path_str = " -> ".join(k)
            print(f"#{upd_rank[k]+1}: Q={upd_q[k]:.3f}")
            print(f"   {path_str}")


if __name__ == "__main__":
    # Default evidence: ReinforcementCageDisplacement excluded -> weight 1 默认证据：排除加强笼位移 -> 权重 1
    run_with_evidence(
        target_label="PileHiddenDefect",
        evidence_weights={"ReinforcementCageDisplacement": 1},
        episodes=4000,
        max_depth=7,
    )

    # Compare updated results with baseline 将更新后的结果与基线结果进行比较
    compare_results(
        baseline_file="q_paths_PileHiddenDefect.json",
        updated_file="q_paths_PileHiddenDefect_evidence.json",
        top_k=10,
    )
