import numpy as np
# from functools import partial
import tensorflow as tf
# from datetime import datetime
from comet_ml import experiment

experiment = Experiment(project_name="sweep-langmuir-ml", workspace="physicistphil")

n_inputs = 500

with tf.name_scope("data"):
    tf.placeholder(tf.float32, [None, n_inputs], name="X")

# with tf.name_scope("network"):
    # tf.initializers.
