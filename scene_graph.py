import numpy as np
from scipy.spatial import cKDTree

class ObjectNode:
    def __init__(self, object_id, centroid, label):
        self.object_id = object_id
        self.centroid = centroid
        self.label = label
        self.neighbors = []

class SceneGraph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def build_scene_graph(self, objects, k=2):
        for i, obj in enumerate(objects):
            node = ObjectNode(object_id=i, centroid=obj[0], label=obj[1])
            self.add_node(node)

        centroids = np.array([node.centroid for node in self.nodes])
        tree = cKDTree(centroids)

        for node in self.nodes:
            _, indices = tree.query(node.centroid, k=k+1)
            for idx in indices[1:]:
                neighbor = self.nodes[idx]
                node.neighbors.append(neighbor)

