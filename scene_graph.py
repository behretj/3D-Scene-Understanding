import numpy as np
from scipy.spatial import cKDTree

class ObjectNode:
    def __init__(self, object_id, centroid, color, sem_label):
        self.object_id = object_id
        self.centroid = centroid
        self.sem_label = sem_label
        self.color = color
        self.neighbors = []

class SceneGraph:
    def __init__(self):
        self.index = 0
        self.nodes = []

    def add_node(self, obj):
        self.nodes.append(ObjectNode(object_id=self.index, centroid=obj[0], color=obj[1], sem_label=obj[2]))
        self.index += 1

    def build_scene_graph(self, objects, k=2):
        for obj in objects:
            self.add_node(obj)

        # TODO: check whether the k-nearest neighbors are correct
        centroids = np.array([node.centroid for node in self.nodes])
        tree = cKDTree(centroids)

        for node in self.nodes:
            _, indices = tree.query(node.centroid, k=k+1)
            for idx in indices[1:]:
                neighbor = self.nodes[idx]
                node.neighbors.append(neighbor)

