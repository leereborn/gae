from initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

        self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class AttentiveGraphConvolution(Layer):
    def __init__(self, input_dim, output_dim, adj, in_drop=0., attn_drop=0., feat_drop=0., act=tf.nn.relu, **kwargs):
        super(AttentiveGraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights") # Why not use tf official glorot init?
            self.vars['attn_self'] = tf.get_variable(name = "attn_self", shape=(output_dim,1),initializer=tf.glorot_uniform_initializer)
            self.vars['attn_neigh'] = tf.get_variable(name = "attn_neigh", shape=(output_dim,1),initializer=tf.glorot_uniform_initializer)
        self.in_drop = in_drop
        self.attn_drop = attn_drop
        self.feat_drop = feat_drop
        self.adj = adj
        self.act = act

        self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.in_drop)
        x = tf.matmul(x, self.vars['weights'])
        # attention
        attn_self = tf.matmul(x, self.vars['attn_self'])
        attn_neigh = tf.matmul(x,self.vars['attn_neigh'])
        attn_coef = attn_self + tf.transpose(attn_neigh)
        attn_coef = tf.nn.leaky_relu(attn_coef)
        dense_adj = tf.sparse.to_dense(self.adj)
        mask = -10e9 * (1.0 - dense_adj)
        attn_coef += mask
        attn_coef = tf.nn.softmax(attn_coef)
        # attn dropout
        attn_coef = tf.nn.dropout(attn_coef,rate=self.attn_drop)
        tf.summary.histogram('hidden2_edge_weights',attn_coef)
        x = tf.nn.dropout(x,rate=self.feat_drop)

        x = tf.matmul(attn_coef, x)
        outputs = self.act(x)
        return outputs

class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

        self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class AttentiveGraphConvolutionSparse(Layer):
    def __init__(self, input_dim, output_dim, adj, features_nonzero, in_drop=0., attn_drop=0., feat_drop=0., act=tf.nn.relu, **kwargs):
        super(AttentiveGraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
            #self.vars['attn_weights'] = weight_variable_glorot(output_dim, 1, name="attn_weights")
            self.vars['attn_self'] = tf.get_variable(name = "attn_self", shape=(output_dim,1),initializer=tf.glorot_uniform_initializer)
            self.vars['attn_neigh'] = tf.get_variable(name = "attn_neigh", shape=(output_dim,1),initializer=tf.glorot_uniform_initializer)
            self.vars['attn_coef'] = tf.get_variable(name = "attn_coef", shape=(input_dim,input_dim))
            #print(self.vars['attn_self'])
            #import pdb;pdb.set_trace()
        self.in_drop = in_drop
        self.attn_drop = attn_drop
        self.feat_drop = feat_drop
        self.adj = adj
        self.act = act
        self.issparse = True # What is the purpose?
        self.features_nonzero = features_nonzero

        self._log_vars() # write summary

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.in_drop, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights']) # returns a dense matrix
        # attention
        attn_self = tf.matmul(x, self.vars['attn_self'])
        attn_neigh = tf.matmul(x,self.vars['attn_neigh'])
        attn_coef = attn_self + tf.transpose(attn_neigh)
        attn_coef = tf.nn.leaky_relu(attn_coef)
        dense_adj = tf.sparse.to_dense(self.adj) # why need not reorder here but in gcmc
        mask = -10e9 * (1.0 - dense_adj)
        attn_coef += mask
        attn_coef = tf.nn.softmax(attn_coef)
        # attn dropout
        attn_coef = tf.nn.dropout(attn_coef,rate=self.attn_drop)
        #print(attn_coef)
        #import pdb;pdb.set_trace()
        #tf.summary.histogram('hidden1_edge_weights',attn_coef)
        x = tf.nn.dropout(x,rate=self.feat_drop)
        
        x = tf.matmul(attn_coef, x)
        #print(attn_coef)
        #import pdb;pdb.set_trace()
        outputs = self.act(x)
        return outputs

class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs
