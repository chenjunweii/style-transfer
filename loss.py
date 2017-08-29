import mxnet as mx


def mse(a, b):
    
    """
        
    Mean Square Error

    """

    return mx.symbol.sum(data = 0.5 * (a - b) ** 2)


def correlation(style, noise, kwarg):

    """

    Gram Matrix

    """
    
    b, c, h, w = style.infer_shape(**kwarg)[1][0]

    s = style.reshape((c, h * w))

    n = noise.reshape((c, h * w))

    s = mx.symbol.dot(s, mx.symbol.transpose(data = s))
    
    n = mx.symbol.dot(n, mx.symbol.transpose(data = n))
    
    #print s.infer_shape(**kwarg)[1][0]

    return mx.symbol.sum(data = (n - s) ** 2) / (4 * (c**2) * (h*w) ** 2)
