import sys
import caffe

from caffe import layers as L
from caffe import params as P

class EdLeNet( object ):
    def __init__( self, lmdb_train, lmdb_test, num_output ):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.class_num = num_output
    
    def EdLeNet( self, batch_size ):
        n = caffe.NetSpec()

        # data
        n.data, n.label = L.Data(
            source = self.train_data,
            backend = P.Data.LMDB,
            batch_size = batch_size,
            ntop = 2,
            transform_param = dict( scale = 1./256, mirror=False )
        )

        # conv1
        n.conv1 = L.Convolution( n.data,
        kernel_size = 3, num_output = 32, stride=1,
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
        kernel_size = 3, num_output = 64, stride=1,
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
        kernel_size=3, num_output=128, stride=1,
        weight_filler=dict( type='xavier'),
        bias_filler = dict( type='constant')
        )
        # relu3
        n.relu3 = L.ReLU( n.conv3, in_place=True)
        # pool3
        n.pool3 = L.Pooling( n.relu3,
        pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.ip1 = L.InnerProduct( n.pool3, num_output=120,
        weight_filler=dict(type='xavier'),
        bias_filler = dict(type='constant'))
        
        n.ip2 = L.InnerProduct( n.ip1, num_output=84,
        weight_filler=dict( type='xavier'),
        bias_filler=dict(type='constant'))

        n.ip3 = L.InnerProduct( n.ip2, 
        num_output=self.class_num,
        weight_filler=dict( type='xavier'),
        bias_filler = dict( type='constant'))

        n.loss = L.SoftmaxWithLoss( n.ip3, n.label)
        return n.to_proto()
if __name__ == '__main__':
    l = EdLeNet('train_lmdb','validation_lmdb',43)
    print l.EdLeNet(100)



        





