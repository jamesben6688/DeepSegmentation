import tensorflow as tf

class AutoAugment(object):

    def __init__(self, sample):

        # random flips
        sample = self.random_apply(self.random_flips, sample, p=0.5)

        # random brightness/contrast
        sample = self.random_apply(self.random_brightness_and_contrast, sample, p=0.5)
        
        # random noise (Gaussian)

        # random filter (sobel)

        # random resolution

        # V00 <--> V01 of the same knee

        return sample

    def random_flips(self, x):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)

        return x

    def random_brightness_and_contrast(self, x, s=1):
        x = tf.image.random_brightness(x, max_delta=0.8 * s)
        x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)

        return x

    def random_apply(self, function, sample, p):
        return tf.cond(tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), 
                       tf.cast(p, tf.float32)),
                       lambda: function(sample),
                       lambda: sample)
