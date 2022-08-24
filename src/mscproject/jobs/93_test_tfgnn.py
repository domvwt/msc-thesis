import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner

graph_schema = tfgnn.read_schema("/tmp/graph_schema.pbtxt")
gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

# len(train_ds_provider.get_dataset(...)) == 8191.
train_ds_provider = runner.TFRecordDatasetProvider(file_pattern="...")
# len(valid_ds_provider.get_dataset(...)) == 1634.
valid_ds_provider = runner.TFRecordDatasetProvider(file_pattern="...")

# Extract labels from the graph context, importantly: this lambda matches
# the `GraphTensorProcessorFn` protocol (see below).
extract_labels = lambda gt: gt, gt.context["label"]

# Use `embedding` feature as the only node feature.
initial_node_states = lambda node_set, node_set_name: node_set["embedding"]
# This `tf.keras.layers.Layer` matches the `GraphTensorProcessorFn` protocol.
map_features = tfgnn.keras.layers.MapFeatures(node_sets_fn=initial_node_states)

# Binary classification by the root node.
task = runner.RootNodeBinaryClassification(node_set_name="nodes")

trainer = runner.KerasTrainer(
    strategy=tf.distribute.TPUStrategy(...),
    model_dir="...",
    steps_per_epoch=8191 // 128,  # global_batch_size == 128
    validation_per_epoch=2,
    validation_steps=1634 // 128,
)  # global_batch_size == 128

runner.run(
    train_ds_provider=train_ds_provider,
    train_padding=runner.FitOrSkipPadding(gtspec, train_ds_provider),
    # simple_gnn is a function: Callable[[tfgnn.GraphTensorSpec], tf.keras.Model].
    # Where the returned model both takes and returns a scalar `GraphTensor` for
    # its inputs and outputs.
    model_fn=simple_gnn,
    optimizer_fn=tf.keras.optimizers.Adam,
    epochs=4,
    trainer=trainer,
    task=task,
    gtspec=gtspec,
    global_batch_size=128,
    # Extract any labels before lossing them via `MapFeatures`!
    feature_processors=[extract_labels, map_features],
    valid_ds_provider=valid_ds_provider,
)
