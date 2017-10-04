import sys
import caffe

from caffe import layers as L
from caffe import params as P

class EdLeNet2( object ):
    def __init__( self, lmdb_train, lmdb_test, num_output, batch_size ):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.class_num = num_output
        self.batch_size = batch_size
    
    def EdLeNet( self ):
        n = caffe.NetSpec()

        # data
        n.data, n.label = L.Data(
            source = self.train_data,
            backend = P.Data.LMDB,
            batch_size = self.batch_size,
            ntop = 2,
            transform_param = dict( scale = 1./256, mirror=False )
        )

        # conv1
        n.conv1 = L.Convolution( n.data,
        kernel_size = 5, num_output = 32, stride=1, pad = 2,
        weight_filler = dict( type='xavier' ),
        bias_filler = dict( type = 'constant' )
        )
        # relu1
        n.relu1 = L.ReLU( n.conv1, in_place=True)
        # pool1
        n.pool1 = L.Pooling( n.relu1,
        pool=P.Pooling.MAX, kernel_size=2, stride=2 )

        # conv2
        n.conv2 = L.Convolution( n.pool1,
        kernel_size = 5, num_output = 64, stride=1, pad = 2,
        weight_filler = dict( type='xavier' ),
        bias_filler = dict( type = 'constant' )
        )
        # relu2
        n.relu2 = L.ReLU( n.conv2, in_place=True)
        # pool2
        n.pool2 = L.Pooling( n.relu2,
        pool=P.Pooling.MAX, kernel_size=2, stride=2 )

        # conv3
        n.conv3 = L.Convolution( n.pool2,
        kernel_size=5, num_output=128, stride=1, pad = 2,
        weight_filler=dict( type='xavier'),
        bias_filler = dict( type='constant')
        )
        # relu3
        n.relu3 = L.ReLU( n.conv3, in_place=True)
        # pool3
        n.pool3 = L.Pooling( n.relu3,
        pool=P.Pooling.MAX, kernel_size=2, stride=2)

        print type( n.pool3 )

        
        n.pool_reshape1 = L.Pooling( n.pool1, 
        pool = P.Pooling.MAX, kernel_size=4, stride=4 )
        # [1 32 4 4] -> [1 512 1 1]
        n.pool1_flatten = L.Reshape( n.pool_reshape1, reshape_param={'shape':{'dim': [ self.batch_size, 512, 1, 1]}} )

        n.pool_reshape2 = L.Pooling( n.pool2,
        pool=P.Pooling.MAX, kernel_size=2, stride=2)
        # [1 64 4 4] -> [1 1024 1 1]
        n.pool2_flatten = L.Reshape( n.pool_reshape2, reshape_param={'shape':{'dim': [self.batch_size, 1024, 1, 1]}} )

        # [1 128 4 4] -> [1 2048 1 1]
        n.pool3_flatten = L.Reshape( n.pool3, reshape_param={'shape':{'dim': [self.batch_size, 2048, 1, 1]}} )

        concat_bottom_layers = [n.pool1_flatten,n.pool2_flatten,n.pool3_flatten ]
        n.concatLayer = L.Concat( *concat_bottom_layers )

        n.ip1 = L.InnerProduct( n.concatLayer , num_output=1024,
        weight_filler=dict(type='xavier'),
        bias_filler = dict(type='constant'))
        
        n.ip2 = L.InnerProduct( n.ip1, 
        num_output=self.class_num,
        weight_filler=dict( type='xavier'),
        bias_filler = dict( type='constant'))

        n.loss = L.SoftmaxWithLoss( n.ip2, n.label)
        return n.to_proto()
if __name__ == '__main__':
    l = EdLeNet2('train_lmdb','validation_lmdb',43, 100)
    print l.EdLeNet()



        





