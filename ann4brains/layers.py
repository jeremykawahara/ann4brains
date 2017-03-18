from caffe import layers as L
from caffe import params as P


def full_connect(bottom, num_output,
                 weight_filler=dict(type="xavier"),
                 bias_filler=dict(type="constant", value=0),
                 param=[dict(lr_mult=1, decay_mult=1),  # weight learning rate parameters
                        dict(lr_mult=2, decay_mult=0)]  # bias learning rate parameters
                 ):
    fc = L.InnerProduct(bottom, num_output=num_output,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler,
                        param=param)
    return fc


def e2n_conv(bottom, num_output, kernel_h, kernel_w,
             weight_filler=dict(type="xavier"),
             bias_filler=dict(type="constant", value=0),
             param=[dict(lr_mult=1, decay_mult=1),  # weight learning rate parameters
                    dict(lr_mult=2, decay_mult=0)]  # bias learning rate parameters
             ):
    """Edge-to-Node convolution.

    This is implemented only as a 1 x d rather than combined with d x 1,
    since our tests did not show a consistent improvement with them combined.
    """

    # 1xL convolution.
    conv_1xd = L.Convolution(bottom, num_output=num_output, stride=1,
                             kernel_h=1, kernel_w=kernel_w,
                             weight_filler=weight_filler, bias_filler=bias_filler,
                             param=param)
    return conv_1xd


def e2e_conv(bottom, num_output, kernel_h, kernel_w,
             weight_filler=dict(type="xavier"),
             bias_filler=dict(type="constant", value=0),
             param=[dict(lr_mult=1, decay_mult=1),  # weight learning rate parameters
                    dict(lr_mult=2, decay_mult=0)]  # bias learning rate parameters
             ):
    """Implementation of the e2e filter."""

    # kernel_h x 1 convolution.
    conv_dx1 = L.Convolution(bottom, num_output=num_output, stride=1,
                             kernel_h=kernel_h, kernel_w=1,
                             weight_filler=weight_filler, bias_filler=bias_filler,
                             param=param)

    # 1 x kernel_w convolution.
    conv_1xd = L.Convolution(bottom, num_output=num_output, stride=1,
                             kernel_h=1, kernel_w=kernel_w,
                             weight_filler=weight_filler, bias_filler=bias_filler,
                             param=param)

    # Concat all the responses together.
    # For dx1, produce a dxd matrix.
    concat_dx1_dxd = L.Concat(*[conv_dx1] * kernel_w, concat_param=dict(axis=2))

    # For 1xd, produce a dxd matrix.
    concat_1xd_dxd = L.Concat(*[conv_1xd] * kernel_h, concat_param=dict(axis=3))

    # Sum the dxd matrices together element-wise.
    sum_dxd = L.Eltwise(concat_dx1_dxd, concat_1xd_dxd, eltwise_param=dict(operation=P.Eltwise.SUM))

    return sum_dxd