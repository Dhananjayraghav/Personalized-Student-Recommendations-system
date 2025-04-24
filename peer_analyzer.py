import networkx as nx
import numpy as np
from community import community_louvain
from typing import Dict, List, Tuple


class PeerAnalyzer:
    def __init__(self, interactions):
        self.graph = self.build_social_graph(interactions)
        self.communities = self.detect_communities()

    def build_social_graph(self, interactions) -> nx.Graph:
        G = nx.Graph()

        # Add nodes (students)
        student_ids = set(interactions['student_id'])
        G.add_nodes_from(student_ids)

        # Add edges based on shared resources
        resource_students = interactions.groupby('resource_id')['student_id'].apply(set)

        for students in resource_students:
            students = list(students)
            for i in range(len(students)):
                for j in range(i + 1, len(students)):
                    if G.has_edge(students[i], students[j]):
                        G[students[i]][students[j]]['weight'] += 1
                    else:
                        G.add_edge(students[i], students[j], weight=1)

        return G

    def detect_communities(self) -> Dict[int, int]:
        return community_louvain.best_partition(self.graph)

    def get_social_recommendations(self, student_id: int, top_n: int = 5) -> List[int]:
        if student_id not in self.graph:
            return []

        # Get peers in the same community
        community_id = self.communities[student_id]
        peers = [n for n in self.graph.nodes if self.communities[n] == community_id and n != student_id]

        # Rank peers by connection strength
        peer_scores = []
        for peer in peers:
            if self.graph.has_edge(student_id, peer):
                peer_scores.append((peer, self.graph[student_id][peer]['weight']))
            else:
                peer_scores.append((peer, 0))

        # Sort by connection strength
        peer_scores.sort(key=lambda x: x[1], reverse=True)

        return [peer for peer, _ in peer_scores[:top_n]]

    def get_community_resources(self, student_id: int, resource_history: set, top_n: int = 5) -> List[int]:
        if student_id not in self.communities:
            return []

        community_id = self.communities[student_id]
        community_members = [n for n, c in self.communities.items() if c == community_id]

        # Get popular resources in community not seen by student
        community_resources = {}
        for member in community_members:
            member_resources = set(self.interactions[self.interactions['student_id'] == member]['resource_id'])
            for res in member_resources:
                if res not in resource_history:
                    community_resources[res] = community_resources.get(res, 0) + 1

        return sorted(community_resources.items(), key=lambda x: x[1], reverse=True)[:top_n]