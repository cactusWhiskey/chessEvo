from evolution import tensor_node
from evolution.tensor_network import TensorNetwork
import tensorflow as tf

tn = TensorNetwork([(10,)], [2])
graph = tn.graph
dense = tensor_node.DenseNode(graph, 5)
tn.insert_node(dense, 0)

inputs = tn.all_nodes[tn.input_ids[0]](tn.all_nodes)
outputs = tn.all_nodes[tn.output_ids[0]](tn.all_nodes)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()