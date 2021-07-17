import numpy as np
import tensorflow as tf
import spatial_transformer


class TVNet(object):
    GRAD_IS_ZERO = 1e-12

    def __init__(self):
        pass

    def grey_scale_image(self, x):
        assert len(x.shape) == 4
        assert x.shape[-1].value == 3, 'number of channels must be 3 (i.e. RGB)'

        ker_init = tf.constant_initializer([[0.114], [0.587], [0.299]])
        grey_x = tf.layers.conv2d(x, 1, [1, 1], padding='same',
                                  kernel_initializer=ker_init, use_bias=False, trainable=False)

        return tf.floor(grey_x)

    def normalize_images(self, x1, x2):
        reduction_axes = [i for i in xrange(1, len(x1.shape))]
        min_x1 = tf.reduce_min(x1, axis=reduction_axes)
        max_x1 = tf.reduce_max(x1, axis=reduction_axes)

        min_x2 = tf.reduce_min(x2, axis=reduction_axes)
        max_x2 = tf.reduce_max(x2, axis=reduction_axes)

        min_val = tf.minimum(min_x1, min_x2)
        max_val = tf.maximum(max_x1, max_x2)

        den = max_val - min_val

        expand_dims = [-1 if i == 0 else 1 for i in xrange(len(x1.shape))]
        min_val_ex = tf.reshape(min_val, expand_dims)
        den_ex = tf.reshape(den, expand_dims)

        x1_norm = tf.where(den > 0, 255. * (x1 - min_val_ex) / den_ex, x1)
        x2_norm = tf.where(den > 0, 255. * (x2 - min_val_ex) / den_ex, x2)

        return x1_norm, x2_norm

    def gaussian_smooth(self, x):
        assert len(x.shape) == 4
        ker_init = tf.constant_initializer([[0.000874, 0.006976, 0.01386, 0.006976, 0.000874],
                                            [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
                                            [0.01386, 0.110656, 0.219833, 0.110656, 0.01386],
                                            [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
                                            [0.000874, 0.006976, 0.01386, 0.006976, 0.000874]])
        smooth_x = tf.layers.conv2d(x, x.shape[-1].value, [5, 5], padding='same',
                                    kernel_initializer=ker_init, use_bias=False, trainable=False)

        return smooth_x

    def warp_image(self, x, u, v):
        assert len(x.shape) == 4
        assert len(u.shape) == 3
        assert len(v.shape) == 3
        u = u / x.shape[2].value * 2
        v = v / x.shape[1].value * 2

        delta = tf.concat(axis=1, values=[u, v])
        return spatial_transformer.transformer(x, delta, (x.shape[-3].value, x.shape[-2].value))

    def centered_gradient(self, x, name):
        assert len(x.shape) == 4

        with tf.variable_scope('centered_gradient'):
            x_ker_init = tf.constant_initializer([[-0.5, 0, 0.5]])
            diff_x = tf.layers.conv2d(x, x.shape[-1].value, [1, 3], padding='same',
                                      kernel_initializer=x_ker_init, use_bias=False, name=name + '_diff_x',
                                      trainable=False)

            y_ker_init = tf.constant_initializer([[-0.5], [0], [0.5]])
            diff_y = tf.layers.conv2d(x, x.shape[-1].value, [3, 1], padding='same',
                                      kernel_initializer=y_ker_init, use_bias=False, name=name + '_diff_y',
                                      trainable=False)

            # refine the boundary
            first_col = 0.5 * (tf.slice(x, [0, 0, 1, 0], [-1, x.shape[1].value, 1, x.shape[3].value]) -
                               tf.slice(x, [0, 0, 0, 0], [-1, x.shape[1].value, 1, x.shape[3].value]))

            last_col = 0.5 * (
                tf.slice(x, [0, 0, x.shape[2].value - 1, 0], [-1, x.shape[1].value, 1, x.shape[3].value]) -
                tf.slice(x, [0, 0, x.shape[2].value - 2, 0], [-1, x.shape[1].value, 1, x.shape[3].value]))
            diff_x_valid = tf.slice(diff_x, begin=[0, 0, 1, 0],
                                    size=[-1, x.shape[1].value, x.shape[2].value - 2, x.shape[3].value])
            diff_x = tf.concat(axis=2, values=[first_col, diff_x_valid, last_col])

            first_row = 0.5 * (tf.slice(x, [0, 1, 0, 0], [-1, 1, x.shape[2].value, x.shape[3].value]) -
                               tf.slice(x, [0, 0, 0, 0], [-1, 1, x.shape[2].value, x.shape[3].value]))
            last_row = 0.5 * (
                tf.slice(x, [0, x.shape[1].value - 1, 0, 0], [-1, 1, x.shape[2].value, x.shape[3].value]) -
                tf.slice(x, [0, x.shape[1].value - 2, 0, 0], [-1, 1, x.shape[2].value, x.shape[3].value]))
            diff_y_valid = tf.slice(diff_y, begin=[0, 1, 0, 0],
                                    size=[-1, x.shape[1].value - 2, x.shape[2].value, x.shape[3].value])
            diff_y = tf.concat(axis=1, values=[first_row, diff_y_valid, last_row])

        return diff_x, diff_y

    def forward_gradient(self, x, name):
        assert len(x.shape) == 4

        with tf.variable_scope('forward_gradient'):
            x_ker_init = tf.constant_initializer([[-1, 1]])
            diff_x = tf.layers.conv2d(x, x.shape[-1].value, [1, 2], padding='same',
                                      kernel_initializer=x_ker_init, use_bias=False, name=name + '_diff_x',
                                      trainable=True)

            y_ker_init = tf.constant_initializer([[-1], [1]])
            diff_y = tf.layers.conv2d(x, x.shape[-1].value, [2, 1], padding='same',
                                      kernel_initializer=y_ker_init, use_bias=False, name=name + '_diff_y',
                                      trainable=True)

            # refine the boundary
            diff_x_valid = tf.slice(diff_x, begin=[0, 0, 0, 0],
                                    size=[-1, x.shape[1].value, x.shape[2].value - 1, x.shape[3].value])
            last_col = tf.zeros([tf.shape(x)[0], x.shape[1].value, 1, x.shape[3].value], dtype=tf.float32)
            diff_x = tf.concat(axis=2, values=[diff_x_valid, last_col])

            diff_y_valid = tf.slice(diff_y, begin=[0, 0, 0, 0],
                                    size=[-1, x.shape[1].value - 1, x.shape[2].value, x.shape[3].value])
            last_row = tf.zeros([tf.shape(x)[0], 1, x.shape[2].value, x.shape[3].value], dtype=tf.float32)
            diff_y = tf.concat(axis=1, values=[diff_y_valid, last_row])

        return diff_x, diff_y
    
    def forward_gradient_forloss(self, x, name):
        assert len(x.shape) == 4

        with tf.variable_scope('forward_gradient'):
            x_ker_init = tf.constant_initializer([[-1, 1]])
            diff_x = tf.layers.conv2d(x, x.shape[-1].value, [1, 2], padding='same',
                                      kernel_initializer=x_ker_init, use_bias=False, name=name + '_diff_x',
                                      trainable=True)

            y_ker_init = tf.constant_initializer([[-1], [1]])
            diff_y = tf.layers.conv2d(x, x.shape[-1].value, [2, 1], padding='same',
                                      kernel_initializer=y_ker_init, use_bias=False, name=name + '_diff_y',
                                      trainable=True)

            # refine the boundary
            diff_x_valid = tf.slice(diff_x, begin=[0, 0, 0, 0],
                                    size=[-1, x.shape[1].value, x.shape[2].value - 1, x.shape[3].value])
            last_col = tf.zeros([tf.shape(x)[0], x.shape[1].value, 1, x.shape[3].value], dtype=tf.float32)
            diff_x = tf.concat(axis=2, values=[diff_x_valid, last_col])

            diff_y_valid = tf.slice(diff_y, begin=[0, 0, 0, 0],
                                    size=[-1, x.shape[1].value - 1, x.shape[2].value, x.shape[3].value])
            last_row = tf.zeros([tf.shape(x)[0], 1, x.shape[2].value, x.shape[3].value], dtype=tf.float32)
            diff_y = tf.concat(axis=1, values=[diff_y_valid, last_row])

        return diff_x, diff_y
    
    def divergence(self, x, y, name):
        assert len(x.shape) == 4

        with tf.variable_scope('divergence'):
            x_valid = tf.slice(x, begin=[0, 0, 0, 0],
                               size=[-1, x.shape[1].value, x.shape[2].value - 1, x.shape[3].value])
            first_col = tf.zeros([tf.shape(x)[0], x.shape[1].value, 1, x.shape[3].value], dtype=tf.float32)
            x_pad = tf.concat(axis=2, values=[first_col, x_valid])

            y_valid = tf.slice(y, begin=[0, 0, 0, 0],
                               size=[-1, y.shape[1].value - 1, y.shape[2].value, y.shape[3].value])
            first_row = tf.zeros([tf.shape(y)[0], 1, y.shape[2].value, y.shape[3].value], dtype=tf.float32)
            y_pad = tf.concat(axis=1, values=[first_row, y_valid])

            x_ker_init = tf.constant_initializer([[-1, 1]])
            diff_x = tf.layers.conv2d(x_pad, x.shape[-1].value, [1, 2], padding='same',
                                      kernel_initializer=x_ker_init, use_bias=False, name=name + '_diff_x',
                                      trainable=True)

            y_ker_init = tf.constant_initializer([[-1], [1]])
            diff_y = tf.layers.conv2d(y_pad, y.shape[-1].value, [2, 1], padding='same',
                                      kernel_initializer=y_ker_init, use_bias=False, name=name + '_diff_y',
                                      trainable=True)

        div = diff_x + diff_y
        return div

    def zoom_size(self, height, width, factor):
        new_height = int(float(height) * factor + 0.5)
        new_width = int(float(width) * factor + 0.5)

        return new_height, new_width

    def zoom_image(self, x, new_height, new_width):
        assert len(x.shape) == 4

        delta = tf.zeros((tf.shape(x)[0], 2, new_height * new_width))
        zoomed_x = spatial_transformer.transformer(x, delta, (new_height, new_width))
        return tf.reshape(zoomed_x, [tf.shape(x)[0], new_height, new_width, x.shape[-1].value])

    def dual_tvl1_optic_flow(self, x1, x2, u1, u2,
                             tau=0.25,  # time step
                             lbda=0.15,  # weight parameter for the data term
                             theta=0.3,  # weight parameter for (u - v)^2
                             warps=5,  # number of warpings per scale
                             max_iterations=5  # maximum number of iterations for optimization
                             ):

        l_t = lbda * theta
        taut = tau / theta

        diff2_x, diff2_y = self.centered_gradient(x2, 'x2')

        p11 = p12 = p21 = p22 = tf.zeros_like(x1)

        for warpings in xrange(warps):
            with tf.variable_scope('warping%d' % (warpings,)):
                u1_flat = tf.reshape(u1, (tf.shape(x2)[0], 1, x2.shape[1].value * x2.shape[2].value))
                u2_flat = tf.reshape(u2, (tf.shape(x2)[0], 1, x2.shape[1].value * x2.shape[2].value))

                x2_warp = self.warp_image(x2, u1_flat, u2_flat)
                x2_warp = tf.reshape(x2_warp, tf.shape(x2))

                diff2_x_warp = self.warp_image(diff2_x, u1_flat, u2_flat)
                diff2_x_warp = tf.reshape(diff2_x_warp, tf.shape(diff2_x))

                diff2_y_warp = self.warp_image(diff2_y, u1_flat, u2_flat)
                diff2_y_warp = tf.reshape(diff2_y_warp, tf.shape(diff2_y))

                diff2_x_sq = tf.square(diff2_x_warp)
                diff2_y_sq = tf.square(diff2_y_warp)

                grad = diff2_x_sq + diff2_y_sq + self.GRAD_IS_ZERO

                rho_c = x2_warp - diff2_x_warp * u1 - diff2_y_warp * u2 - x1

                for ii in xrange(max_iterations):
                    with tf.variable_scope('iter%d' % (ii,)):
                        rho = rho_c + diff2_x_warp * u1 + diff2_y_warp * u2 + self.GRAD_IS_ZERO;

                        masks1 = rho < -l_t * grad
                        d1_1 = tf.where(masks1, l_t * diff2_x_warp, tf.zeros_like(diff2_x_warp))
                        d2_1 = tf.where(masks1, l_t * diff2_y_warp, tf.zeros_like(diff2_y_warp))

                        masks2 = rho > l_t * grad
                        d1_2 = tf.where(masks2, -l_t * diff2_x_warp, tf.zeros_like(diff2_x_warp))
                        d2_2 = tf.where(masks2, -l_t * diff2_y_warp, tf.zeros_like(diff2_y_warp))

                        masks3 = (~masks1) & (~masks2) & (grad > self.GRAD_IS_ZERO)
                        d1_3 = tf.where(masks3, -rho / grad * diff2_x_warp, tf.zeros_like(diff2_x_warp))
                        d2_3 = tf.where(masks3, -rho / grad * diff2_y_warp, tf.zeros_like(diff2_y_warp))

                        v1 = d1_1 + d1_2 + d1_3 + u1
                        v2 = d2_1 + d2_2 + d2_3 + u2

                        u1 = v1 + theta * self.divergence(p11, p12, 'div_p1')
                        u2 = v2 + theta * self.divergence(p21, p22, 'div_p2')

                        u1x, u1y = self.forward_gradient(u1, 'u1')
                        u2x, u2y = self.forward_gradient(u2, 'u2')

                        p11 = (p11 + taut * u1x) / (
                            1.0 + taut * tf.sqrt(tf.square(u1x) + tf.square(u1y) + self.GRAD_IS_ZERO));
                        p12 = (p12 + taut * u1y) / (
                            1.0 + taut * tf.sqrt(tf.square(u1x) + tf.square(u1y) + self.GRAD_IS_ZERO));
                        p21 = (p21 + taut * u2x) / (
                            1.0 + taut * tf.sqrt(tf.square(u2x) + tf.square(u2y) + self.GRAD_IS_ZERO));
                        p22 = (p22 + taut * u2y) / (
                            1.0 + taut * tf.sqrt(tf.square(u2x) + tf.square(u2y) + self.GRAD_IS_ZERO));

        return u1, u2, rho

    def tvnet_flow(self, x1, x2,
                    tau=0.25,  # time step
                    lbda=0.15,  # weight parameter for the data term
                    theta=0.3,  # weight parameter for (u - v)^2
                    warps=5,  # number of warpings per scale
                    zfactor=0.5,  # factor for building the image piramid
                    max_scales=5,  # maximum number of scales for image piramid
                    max_iterations=5  # maximum number of iterations for optimization
                    ):

        for i in xrange(len(x1.shape)):
            assert x1.shape[i].value == x2.shape[i].value

        zfactor = np.float32(zfactor)

        height = x1.shape[-3].value
        width = x1.shape[-2].value

        n_scales = 1 + np.log(np.sqrt(height ** 2 + width ** 2) / 4.0) / np.log(1 / zfactor);
        n_scales = min(n_scales, max_scales)
        # n_scales = 1
        with tf.variable_scope('tvl1_flow'):
            grey_x1 = self.grey_scale_image(x1)
            grey_x2 = self.grey_scale_image(x2)
            norm_imgs = self.normalize_images(grey_x1, grey_x2)

            smooth_x1 = self.gaussian_smooth(norm_imgs[0])
            smooth_x2 = self.gaussian_smooth(norm_imgs[1])
            for ss in xrange(n_scales - 1, -1, -1):
                with tf.variable_scope('scale%d' % ss):
                    down_sample_factor = zfactor ** ss
                    down_height, down_width = self.zoom_size(height, width, down_sample_factor)

                    if ss == n_scales - 1:
                        u1 = tf.get_variable('u1', shape=[1, down_height, down_width, 1], dtype=tf.float32,
                                             initializer=tf.zeros_initializer)
                        u2 = tf.get_variable('u2', shape=[1, down_height, down_width, 1], dtype=tf.float32,
                                             initializer=tf.zeros_initializer)
                        u1 = tf.tile(u1, [tf.shape(smooth_x1)[0], 1, 1, 1])
                        u2 = tf.tile(u2, [tf.shape(smooth_x1)[0], 1, 1, 1])

                    down_x1 = self.zoom_image(smooth_x1, down_height, down_width)
                    down_x2 = self.zoom_image(smooth_x2, down_height, down_width)

                    u1, u2, rho = self.dual_tvl1_optic_flow(down_x1, down_x2, u1, u2,
                                                            tau=tau, lbda=lbda, theta=theta, warps=warps,
                                                            max_iterations=max_iterations)

                    if ss == 0:
                        return u1, u2, rho

                    up_sample_factor = zfactor ** (ss - 1)
                    up_height, up_width = self.zoom_size(height, width, up_sample_factor)
                    u1 = self.zoom_image(u1, up_height, up_width) / zfactor
                    u2 = self.zoom_image(u2, up_height, up_width) / zfactor

    def get_loss(self, x1, x2,
                 tau=0.25,  # time step
                 lbda=0.15,  # weight parameter for the data term
                 theta=0.3,  # weight parameter for (u - v)^2
                 warps=5,  # number of warpings per scale
                 zfactor=0.5,  # factor for building the image piramid
                 max_scales=5,  # maximum number of scales for image piramid
                 max_iterations=5  # maximum number of iterations for optimization
                 ):

        u1, u2, rho = self.tvnet_flow(x1, x2,
                                      tau=tau, lbda=lbda, theta=theta, warps=warps,
                                      zfactor=zfactor, max_scales=max_scales,
                                      max_iterations=max_iterations)

        # computing loss
        u1x, u1y = self.forward_gradient_forloss(u1, 'u1')
        u2x, u2y = self.forward_gradient_forloss(u2, 'u2')


        u1_flat = tf.reshape(u1, (tf.shape(x2)[0], 1, x2.shape[1].value * x2.shape[2].value))
        u2_flat = tf.reshape(u2, (tf.shape(x2)[0], 1, x2.shape[1].value * x2.shape[2].value))

        x2_warp = self.warp_image(x2, u1_flat, u2_flat)
        x2_warp = tf.reshape(x2_warp, tf.shape(x2))
        loss = lbda * tf.reduce_mean(tf.abs(x2_warp - x1)) + tf.reduce_mean(
            tf.abs(u1x) + tf.abs(u1y) + tf.abs(u2x) + tf.abs(u2y))
        return loss, u1, u2
