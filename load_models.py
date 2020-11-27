import tensorflow as tf


# Estimate plasma parameters from a sweep using an attention-based model.
# This assumes a Maxwellian distribution function and estimates the density, plasma potential,
#   and electron temperature.
# Parameters:
#   sweeps: voltage, current pairs of the sweeps you want to evaluate. They should have shape of
#       [-1, 512] where the first 256 are the voltage sweep (no flat bits) and the last 256 are the
#       corresponding current values.
#   model_path: which model you would like to use.
#   len_sweep: the length of the sweeps you're evaluating. Right now only 256 is supported.
# Output:
#   signal_theory: the output sweep based on the estimated plasma parameters. Shape [-1, 256]
#   signal_phys: the estimated plasma parameters. Shape [-1, 3]
#   signal_attention_mask: the attention weight for each part of the sweep. Shape [-1, 256]
def estimate_params_via_attention(sweeps, model_path="models/attn_best.ckpt", len_sweep=256):
    # 20200408214310 = best model
    # 20200418035205 = synthetic sweeps only
    # model-20200618022940-final = semi-supervised
    # 20200701210418 = ???

    # 20200904202916 is so good for some reason

    # 20201104002643 is what I used for the presentation figures (best of attention?)

    # Clear the graph in case any conflicting tensors were defined earlier.
    tf.compat.v1.reset_default_graph()

    # Code to get tensors by name from a graph (in this case, the default one).
    get_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name
    # Get the input data tensor (X) and the training flag tensor from the model.
    X = tf.compat.v1.placeholder(tf.float32, name="X")
    # Import the graph (i.e., tensors) from the checkpoint metafile.
    model_import = tf.compat.v1.train.import_meta_graph(
        model_path + ".ckpt.meta", input_map={"data/X:0": X})
    training = get_tensor("data/training:0")

    # Try two different names for the theory output tensor for backwards compatibility.
    try:
        theory_output = get_tensor("processed_theory:0")
    except:  # Sorry PEP8 for the bare except.
        theory_output = get_tensor("theory_output_normed:0")
    # Plasma info are the physical plasma parameters.
    plasma_info = get_tensor("plasma_info:0")
    # Mean of our data used to train the model.
    data_mean = get_tensor("pipeline/data_mean:0")
    # Peak-to-peak of the data used to train the model.
    data_ptp = get_tensor("pipeline/data_ptp:0")
    # Attention mask tensors (useful for visualzation of model performance and uncertainty).
    attention_mask = get_tensor("attn/attention_mask:0")

    init = tf.compat.v1.global_variables_initializer()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        init.run()
        # Load the model weights from the checkpoint.
        model_import.restore(sess, model_path + ".ckpt")
        # Evaluate the tensors from the load model.
        num_data_mean, num_data_ptp = sess.run([data_mean, data_ptp])
        (signal_theory, signal_phys, signal_attention_mask
         ) = sess.run([theory_output, plasma_info, attention_mask],
                      feed_dict={training: False, X: (sweeps - num_data_mean) / num_data_ptp})

    # Un-normalize model output.
    signal_theory = signal_theory * num_data_ptp[len_sweep:] + num_data_mean[len_sweep:]

    return signal_theory, signal_phys, signal_attention_mask


# Discrepancy
def estimate_params_via_discrepency(sweeps, model_path, len_sweep=256):
    pass
