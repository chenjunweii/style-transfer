import h5py
import mxnet as mx
import numpy as np

vgg16 = dict()

vgg16['layer_29'] = 'conv5_3'
vgg16['layer_27'] = 'conv5_2'
vgg16['layer_25'] = 'conv5_1'

vgg16['layer_22'] = 'conv4_3'
vgg16['layer_20'] = 'conv4_2'
vgg16['layer_18'] = 'conv4_1'

vgg16['layer_15'] = 'conv3_3'
vgg16['layer_13'] = 'conv3_2'
vgg16['layer_11'] = 'conv3_1'

vgg16['layer_8'] = 'conv2_2'
vgg16['layer_6'] = 'conv2_1'

vgg16['layer_3'] = 'conv1_2'
vgg16['layer_1'] = 'conv1_1'


vgg19 = dict()

vgg19['layer_1'] = 'conv1_1'
vgg19['layer_3'] = 'conv1_2'

vgg19['layer_6'] = 'conv2_1'
vgg19['layer_8'] = 'conv2_2'

vgg19['layer_11'] = 'conv3_1'
vgg19['layer_13'] = 'conv3_2'
vgg19['layer_15'] = 'conv3_3'
vgg19['layer_17'] = 'conv3_4'

vgg19['layer_20'] = 'conv4_1'
vgg19['layer_22'] = 'conv4_2'
vgg19['layer_24'] = 'conv4_3'
vgg19['layer_26'] = 'conv4_4'

vgg19['layer_29'] = 'conv5_1'
vgg19['layer_31'] = 'conv5_2'
vgg19['layer_33'] = 'conv5_3'
vgg19['layer_35'] = 'conv5_4'

def load_keras_pretrained(pretrained, args, ctx, model):
    
    vgg = vgg19 if model == 'vgg19' else vgg16

    with h5py.File(pretrained) as f:

        for k in vgg.keys():

            args['w' + vgg[k]] = mx.nd.array(np.array(f[k]['param_0']), ctx)

            args['b' + vgg[k]] = mx.nd.array(np.array(f[k]['param_1']), ctx)
            
            #grad['w' + maps[k]] = mx.nd.zeros_like(data = args['w' + maps[k]])

            #grad['b' + maps[k]] = mx.nd.zeros_like(data = args['b' + maps[k]])

    return args

def load_mat_pretrained(pretrained, args, ctx):
    
    import scipy.io as io

    mat = io.loadmat(pretrained)


    
if __name__ == '__main__':

    weight_args = dict()

    bias_args = dict()

    #load_pretrained('vgg16_weights.h5', weight_args, bias_args, mx.gpu())
    
    load_mat_pretrained('imagenet-vgg-verydeep-19.mat', weight_args, mx.gpu())

