import mxnet as mx
import h5py


def save(filename, args):

    with h5py.File(filename + '.h5', 'w') as f:

        for a in args.keys():
            
            if not a.startwith('w') and not a.startwith('b'):
                
                f.create_dataset(a, args[a].asnumpy())

def load(filename, device):
    
    ndargs = dict()

    with h5py.File(filename) as f:

        for w in weight.keys():

            ndargs[w] = mx.nd.array(np.array(f[w]), device)
        
        for b in bias.keys():

            ndargs[b] = mx.nd.array(np.array(f[b]), device)

    return ndargs
