import mxnet as mx

def conv(prefix, name, node, data, weight, bias, filter, k = (3,3), s = (1,1), relu = True, pad = (1,1)):

    assert len(k) == 2 and len(s) == 2
    
    if 'w' + name not in weight:

        weight['w' + name] = mx.symbol.Variable('w' + name)
        
        bias['b' + name] = mx.symbol.Variable('b' + name)

    output = mx.symbol.Convolution(name = name,
            data = data,
            weight = weight['w' + name],
            bias = bias['b' + name], 
            num_filter = filter,
            kernel = k,
            stride = s,
            pad = pad)

    node[prefix + name] = mx.symbol.Activation(data = output, act_type = 'relu') if relu else output


def dconv(name, node, data, weight, filter, k = (3,3), s = (1,1), relu = True, pad = (1,1), adj = ()):
    assert len(k) == 2 and len(s) == 2

    weight['w' + name] = mx.symbol.Variable('w' + name)
    
    output = mx.symbol.Deconvolution(data = data,
            weight = weight['w' + name],
            num_filter = filter,
            kernel = k,
            stride = s,
            adj = adj,
            pad = pad)

    node[name] = mx.symbol.Activation(data = output, act_type = 'relu') if relu else output

    
def maxpool(name, node, data, k = (2,2), s = (2,2), pad = (0,0)):

    assert len(k) == 2 and len(s) == 2

    node[name] = mx.symbol.Pooling(name = name,
            data = data,
            pool_type = 'max',
            kernel = k,
            stride = s,
            pad = pad)

def avgpool(name, node, data, k = (2,2), s = (2,2), pad = (0,0)):

    assert len(k) == 2 and len(s) == 2

    node[name] =  mx.symbol.Pooling(name = name,
            data = data,
            pool_type = 'avg',
            kernel = k,
            stride = s,
            pad = pad)

def unpool(name, data, s = 2):

    node[name] =  mx.symbol.Pooling(name = name, data = data, scale = s)

def vgg19(inputs, weight, bias, size, style, content, prefix = ''):

    """

    size = [h x w x c]

    """

    node = dict()

    #print size
    
    conv(prefix, 'conv1_1', node, mx.symbol.Reshape(data = inputs, shape = (-1, size[2], size[0], size[1])), weight, bias, 64)    
    conv(prefix, 'conv1_2', node, node[prefix + 'conv1_1'], weight, bias, 64)
    maxpool('pool1', node, node[prefix + 'conv1_2'])

    conv(prefix, 'conv2_1', node, node['pool1'], weight, bias, 128)
    conv(prefix, 'conv2_2', node, node[prefix + 'conv2_1'], weight, bias, 128)
    maxpool('pool2', node, node[prefix + 'conv2_2'])

    conv(prefix, 'conv3_1', node, node['pool2'], weight, bias, 256)
    conv(prefix, 'conv3_2', node, node[prefix + 'conv3_1'], weight, bias, 256)
    conv(prefix, 'conv3_3', node, node[prefix + 'conv3_2'], weight, bias, 256)
    conv(prefix, 'conv3_4', node, node[prefix + 'conv3_3'], weight, bias, 256)
    maxpool('pool3', node, node[prefix + 'conv3_4'], pad = (1,1))

    conv(prefix, 'conv4_1', node, node['pool3'], weight, bias, 512)
    conv(prefix, 'conv4_2', node, node[prefix + 'conv4_1'], weight, bias, 512)
    conv(prefix, 'conv4_3', node, node[prefix + 'conv4_2'], weight, bias, 512)
    conv(prefix, 'conv4_4', node, node[prefix + 'conv4_3'], weight, bias, 512)
    maxpool('pool4', node, node[prefix + 'conv4_4'], k = (3,3), s = (1,1))

    conv(prefix, 'conv5_1', node, node['pool4'], weight, bias, 512)
    conv(prefix, 'conv5_2', node, node[prefix + 'conv5_1'], weight, bias, 512)
    conv(prefix, 'conv5_3', node, node[prefix + 'conv5_2'], weight, bias, 512)
    conv(prefix, 'conv5_4', node, node[prefix + 'conv5_3'], weight, bias, 512)
    maxpool('pool5', node, node[prefix + 'conv5_4'], k = (3,3), s = (2,2))
    

    #return node, OrderedDict(prefix + s:node[prefix+s] for s in style), {prefix + c:node[prefix+c] for c in content}

    return node, [node[prefix+s] for s in style], [node[prefix+c] for c in content]
def vgg16(inputs, weight, bias, size, style, content, prefix = ''):

    """

    size = [h x w x c]

    """

    node = dict()

    #print size
    
    conv(prefix, 'conv1_1', node, mx.symbol.Reshape(data = inputs, shape = (-1, size[2], size[0], size[1])), weight, bias, 64)
    
    conv(prefix, 'conv1_2', node, node[prefix + 'conv1_1'], weight, bias, 64)

    maxpool('pool1', node, node[prefix + 'conv1_2'])

    conv(prefix, 'conv2_1', node, node['pool1'], weight, bias, 128)
    conv(prefix, 'conv2_2', node, node[prefix + 'conv2_1'], weight, bias, 128)
    maxpool('pool2', node, node[prefix + 'conv2_2'])

    conv(prefix, 'conv3_1', node, node['pool2'], weight, bias, 256)
    conv(prefix, 'conv3_2', node, node[prefix + 'conv3_1'], weight, bias, 256)
    conv(prefix, 'conv3_3', node, node[prefix + 'conv3_2'], weight, bias, 256)
    maxpool('pool3', node, node[prefix + 'conv3_3'], pad = (1,1))

    conv(prefix, 'conv4_1', node, node['pool3'], weight, bias, 512)
    conv(prefix, 'conv4_2', node, node[prefix + 'conv4_1'], weight, bias, 512)
    conv(prefix, 'conv4_3', node, node[prefix + 'conv4_2'], weight, bias, 512)
    maxpool('pool4', node, node[prefix + 'conv4_3'], k = (3,3), s = (1,1))

    conv(prefix, 'conv5_1', node, node['pool4'], weight, bias, 512)
    conv(prefix, 'conv5_2', node, node[prefix + 'conv5_1'], weight, bias, 512)
    conv(prefix, 'conv5_3', node, node[prefix + 'conv5_2'], weight, bias, 512)
    maxpool('pool5', node, node[prefix + 'conv5_3'], k = (3,3), s = (2,2))
    

    #return node, OrderedDict(prefix + s:node[prefix+s] for s in style), {prefix + c:node[prefix+c] for c in content}

    return node, [node[prefix+s] for s in style], [node[prefix+c] for c in content]


def devgg(noise, weight, style, content, prefix = ''):

    node = dict()

    dconv('dconv5_3', node, noise, weight, 512)
    dconv('dconv5_2', node, node['dconv5_3'], weight, 512)
    dconv('dconv5_1', node, node['dconv5_2'], weight, 512)

    dconv('dconv4_3', node, node['dconv5_1'], weight, 512, pad = (0,0))
    dconv('dconv4_2', node, node['dconv4_3'], weight, 512)
    dconv('dconv4_1', node, node['dconv4_2'], weight, 256)

    dconv('dconv3_3', node, node['dconv4_1'], weight, 256, s = (2,2))
    dconv('dconv3_2', node, node['dconv3_3'], weight, 256)
    dconv('dconv3_1', node, node['dconv3_2'], weight, 128)

    dconv('dconv2_2', node, node['dconv3_1'], weight, 128, s = (2,2), pad = (1,1), adj = (1,1))
    dconv('dconv2_1', node, node['dconv2_2'], weight, 64)
    
    dconv('dconv1_2', node, node['dconv2_1'], weight, 64, s = (2,2), pad = (1,1), adj = (1,1))
    dconv('dconv1_1', node, node['dconv1_2'], weight, 3)

    dconv('output', node, node['dconv1_1'], weight, 3)

    return node, {'d' + s:node['d' + s] for s in style}, {'d' + c:node['d' + c] for c in content}
    #return node, [node['d' + s] for s in style], [node['d' + c] for c in content]

