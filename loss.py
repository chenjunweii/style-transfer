import mxnet as mx


def variation(noise):
    

    x = mx.symbol.mean((mx.symbol.slice_axis(noise, axis = 2, begin = 0, end = -1) - mx.symbol.slice_axis(noise, axis = 2, begin = 1, end = None)) ** 2)
    y = mx.symbol.mean((mx.symbol.slice_axis(noise, axis = 1, begin = 0, end = -1) - mx.symbol.slice_axis(noise, axis = 1, begin = 1, end = None)) ** 2)
    
    return 2 * (x + y)



def mse(a, b):
    
    """
        
    Mean Square Error

    """

    return mx.symbol.mean(data = 0.5 * (a - b) ** 2)


def correlation(style, noise, kwarg):

    """

    Gram Matrix

    """
    
    b, c, h, w = style.infer_shape(**kwarg)[1][0]
    
    #print '[{} {} {} {}]'.format(b, c, h, w)

    s = style.reshape((c, h * w))

    n = noise.reshape((c, h * w))

    s = mx.symbol.dot(s, mx.symbol.transpose(data = s))
    
    n = mx.symbol.dot(n, mx.symbol.transpose(data = n))
    
    #print s.infer_shape(**kwarg)[1][0]

    return mx.symbol.sum(data = (n - s) ** 2) / ((2 * c * h * w) ** 2)
