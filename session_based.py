import numpy as np
from collections import defaultdict
from typing import List, Dict, Set


class SessionRecommender:
    def __init__(self, session_data):
        self.session_graph = self._build_session_graph(session_data)
        self.transition_matrix = self._build_transition_matrix()

    def _build_session_graph(self, session_data) -> Dict[int, Dict[int, int]]:
        graph = defaultdict(lambda: defaultdict(int))

        for session in session_data:
            resources = session['resources']
            for i in range(len(resources) - 1):
                current = resources[i]
                next_res = resources[i + 1]
                graph[current][next_res] += 1

        return graph

    def _build_transition_matrix(self) -> Dict[int, List[Tuple[int, float]]]:
        transition_matrix = {}

        for source, targets in self.session_graph.items():
            total = sum(targets.values())
            probabilities = [(target, count / total) for target, count in targets.items()]
            probabilities.sort(key=lambda x: x[1], reverse=True)
            transition_matrix[source] = probabilities

        return transition_matrix

    def get_next_recommendations(self, current_session: List[int], n: int = 5) -> List[int]:
        if not current_session:
            return []

        last_resource = current_session[-1]
        if last_resource not in self.transition_matrix:
            return []

        # Get most likely next resources
        transitions = self.transition_matrix[last_resource]
        return [res for res, _ in transitions[:n]]

    def get_session_cluster_recommendations(self, current_session: List[int],
                                            resource_clusters: Dict[int, int],
                                            n: int = 5) -> List[int]:
        if not current_session:
            return []

        # Find most common cluster in current session
        cluster_counts = defaultdict(int)
        for res in current_session:
            if res in resource_clusters:
                cluster_counts[resource_clusters[res]] += 1

        if not cluster_counts:
            return []

        target_cluster = max(cluster_counts.items(), key=lambda x: x[1])[0]

        # Find resources in target cluster not in current session
        cluster_resources = [res for res, cluster in resource_clusters.items()
                             if cluster == target_cluster and res not in current_session]

        return cluster_resources[:n]