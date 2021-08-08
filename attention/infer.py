import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

class Model():

    def __init__(self, path):
        tf.compat.v1.reset_default_graph()
        save_path = "./saved_models/" + path
        get_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name

        self.X = tf.compat.v1.placeholder(tf.float32, name="X")
        model_import = tf.compat.v1.train.import_meta_graph(save_path + ".ckpt.meta",
                                                            input_map={"data/X:0": self.X})
        self.training = get_tensor("data/training:0")
        self.model_output = get_tensor("loss/model_output:0")
        self.plasma_info = get_tensor("plasma_info:0")
        self.data_mean = get_tensor("pipeline/data_mean:0")
        self.data_ptp = get_tensor("pipeline/data_ptp:0")
        self.attention_mask = get_tensor("attn/attention_mask:0")

        init = tf.compat.v1.global_variables_initializer()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(init)
        model_import.restore(self.sess, save_path + ".ckpt")

    def fit(self, sweep):
        n_inputs = 256

        num_data_mean, num_data_ptp = self.sess.run([self.data_mean, self.data_ptp])
        (signal_model, signal_phys, signal_attention_mask
         ) = self.sess.run([self.model_output, self.plasma_info, self.attention_mask],
                           feed_dict={self.training: False,
                                      self.X: (sweep - num_data_mean) / num_data_ptp})

        signal_model = signal_model * num_data_ptp[n_inputs:] + num_data_mean[n_inputs:]

        return signal_model, signal_phys, signal_attention_mask[:, 0, :, 0]

    def close_session(self):
        self.sess.close()
