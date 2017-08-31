import os
import sys
import cv2
import argparse
import mxnet as mx
import numpy as np
from network import *
from model import *
from loss import *
from utils import *
from pretrained import load_keras_pretrained
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--pretrained', dest = 'pretrained', default = '', type = str)
parser.add_argument('--test', dest = 'test', default = '', type = str)
parser.add_argument('--model', dest = 'model', default = '', type = str)
parser.add_argument('--train', dest = 'isTrain', default = True, type = bool)
parser.add_argument('--device', dest = 'device', default = 'cpu', type = str)
parser.add_argument('--iters', dest = 'iterations', default = 1000, type = int)
parser.add_argument('--lr', dest = 'lr', default = 1e1, type = float)
parser.add_argument('--content', dest = 'content', type = str)
parser.add_argument('--style', dest = 'style', type = str)
parser.add_argument('--noise', dest = 'noise', type = str)
parser.add_argument('--output', dest = 'output', default = 'output', type = str)
parser.add_argument('--sweight', dest = 'sweight', default = 5e2, type = float)
parser.add_argument('--cweight', dest = 'cweight', default = 5e0, type = float)
parser.add_argument('--vweight', dest = 'vweight', default = 1e2, type = float)
parser.add_argument('--chk', dest = 'checkpoint', default = 1000, type = int)


options = parser.parse_args()

"""

Handle parameters

"""

if options.style == None:

    print 'Style Image is not set'
    
    sys.exit()

elif options.content == None:

    print 'Content Image is not set'

    sys.exit()


if not os.path.isdir(options.output):

    os.mkdir(options.output)


device = mx.gpu() if options.device == 'gpu' else mx.cpu()


"""

Create NDArray

"""

npcontent = cv2.imread(options.content, True)

shape = npcontent.shape

npstyle = cv2.imread(options.style, True)

ndcontent = mx.nd.array(preprocess(npcontent.astype(np.float32)), device)

ndstyle = mx.nd.array(preprocess(npstyle.astype(np.float32), shape), device)


ndnoise = mx.nd.array(preprocess(np.random.normal(127, 125, npcontent.shape)), device) if options.noise == None \
        else mx.nd.array(preprocess(cv2.imread(options.noise, True).astype(np.float32)), device)


"""

Create Symbol

"""

style = mx.symbol.Variable('style')

content = mx.symbol.Variable('content')

noise = mx.symbol.Variable('noise')

weight = dict() # weight symbol

bias = dict() # bias symbol


"""

define which layer perform content reconstruction and 
       which layer perform style reconstruction

"""

content_layer = ['conv4_2', 'conv5_2']

style_layer = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

wLs = 1 / float(len(style_layer))

wLc = 1 / float(len(content_layer))

batch_size = 1


"""

Define VGG Graph

"""



style_node_all, style_node, _ = vgg19(style, weight, bias, shape, style_layer, content_layer, 's')

content_node_all, _, content_node = vgg19(content, weight, bias, shape, style_layer, content_layer, 'c')

noise_node_all, noise_style_node, noise_content_node = vgg19(noise, weight, bias, shape, style_layer, content_layer, 'n')



"""

Set Input Shape

"""

content_shape_input = {'content' : shape}

style_shape_input = {'style' : shape}

noise_shape_input = {'noise' : shape}

"""

Infer Shapes of content node and style node

"""

content_shape_all, content_output_shape, _ = content_node_all['cconv5_3'].infer_shape(**content_shape_input)

style_shape_all, style_output_shape, _ = style_node_all['sconv5_3'].infer_shape(**style_shape_input)

noise_shape_all, noise_output_shape, _ = noise_node_all['nconv5_3'].infer_shape(**noise_shape_input)

"""

Get node name

"""

content_name = content_node_all['cconv5_3'].list_arguments()

style_name = style_node_all['sconv5_3'].list_arguments()

noise_name = noise_node_all['nconv5_3'].list_arguments()

"""

Mapping node name to shape

"""

content_shape_arg = dict(zip(content_name, content_shape_all))

style_shape_arg = dict(zip(style_name, style_shape_all))



"""

Create Optimizer

"""

Adam = mx.optimizer.Adam(options.lr)

optimizer = mx.optimizer.get_updater(Adam)


"""

nd args

"""

ndarg = dict()

if options.isTrain:
    
    """

    Loss Function

    """

    content_loss = wLc * sum(mse(content_node[c], noise_content_node[c]) for c in xrange(len(content_layer)))
    
    style_loss = wLs * sum(correlation(style_node[s], noise_style_node[s], style_shape_input) for s in xrange(len(style_layer)))
   
    variation_loss = variation(noise)
    
    loss = mx.symbol.MakeLoss(data = options.cweight * content_loss + options.sweight * style_loss + options.vweight * variation_loss)
    
    if options.pretrained:
        
        ndarg, grad = load_keras_pretrained(options.pretrained, ndarg, device, 'vgg19')
    
    else:

        print 'Pretrained Models is not set properly'

    #grad = dict() 
    
    ndarg['content'] = ndcontent

    ndarg['style'] = ndstyle

    ndarg['noise'] = ndnoise

    grad['noise'] = mx.nd.zeros_like(ndnoise)
    
    exe = loss.bind(device, args = ndarg, args_grad = grad)

    #test = style_node_all['sconv1_1'].bind(device, args = ndarg)
    test = style.bind(device, args = ndarg)
    testc = content_node_all['cconv1_1'].bind(device, args = ndarg)
    idx = loss.list_arguments().index('noise')

    for i in xrange(options.iterations):
            
        exe.forward(is_train = True)

        exe.backward()
        
  #      test.forward(is_train = False)
   #     testc.forward(is_train = False)
        """
        for idx, name in enumerate(loss.list_arguments()):

            if name != 'style' and name != 'content':
                
                optimizer(idx, exe.grad_dict[name], exe.arg_dict[name])
        """
        
        optimizer(idx, exe.grad_dict['noise'], exe.arg_dict['noise'])
        
        if i % options.checkpoint == 0 and i != 0:
            print 'Step {}, Loss {}'.format(i, exe.outputs[0].asnumpy())
            #test_image = np.clip(unprocess(test.outputs[0].asnumpy()[0, 0:3, :, :]), 0, 255).astype(np.uint8)
       #     print 'ndstyle : ', ndstyle.shape
      #      print 'npstyle : ', npstyle.shape
     #       print 'ndcontent : ', ndcontent.shape
    #        print 'npcontent : ', npcontent.shape
        #    test_image = np.clip(unprocess(ndstyle.asnumpy()), 0, 255).astype(np.uint8)        
         #   testc_image = np.clip(unprocess(testc.outputs[0].asnumpy()[0, 0:3, :, :]), 0, 255).astype(np.uint8)
            image = np.clip(unprocess(ndnoise.asnumpy()), 0, 255).astype(np.uint8)

            cv2.imwrite('{}/{}.jpg'.format(options.output, i), image)
          #  cv2.imwrite('{}/test{}.jpg'.format(options.output, i), test_image)
           # cv2.imwrite('{}/testc{}.jpg'.format(options.output, i), testc_image)
