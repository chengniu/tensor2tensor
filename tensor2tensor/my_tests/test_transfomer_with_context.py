import tensorflow as tf
import numpy as np
import tensor2tensor.models.transformer_with_context as transformer_with_context
import tensor2tensor.models.transformer as transformer

hparams = transformer.transformer_base()
hparams.hidden_size = 3
hparams.num_heads = 1
hparams.use_target_space_embedding = False
model = transformer_with_context.TransformerWithContext(hparams)

inputs_context_np = [[[[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]]], [[[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]]]]
inputs_context = tf.convert_to_tensor(inputs_context_np, np.float32)
inputs_np = [[[[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]]], [[[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]]]]
inputs = tf.convert_to_tensor(inputs_np, np.float32)
target_space_id = 0
targets_np = [[[[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]]], [[[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]], [[0.3, 0.2, 0.1]]]]
targets = tf.convert_to_tensor(targets_np)

features = dict()
features["inputs_context"] = inputs_context
features["inputs"] = inputs
features["target_space_id"] = target_space_id
features["targets"] = targets

output = model.body(features)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
x =output.eval(session=sess)
print(x)
