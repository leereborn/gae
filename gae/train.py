from __future__ import division
from __future__ import print_function

import time
import os
import datetime
import sys

# Train on CPU (hide GPU) due to memory constraints
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, accuracy_score

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE, MultiHeadedGAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

from statistics import mean, stdev, pstdev
from synthetic_data_generator import get_synthetic_data
import pickle

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('in_drop', 0., 'Input dropout rate (1 - keep probability).')
flags.DEFINE_float('attn_drop', 0., 'Attention dropout rate (1 - keep probability).')
flags.DEFINE_float('feat_drop', 0., 'Feature dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_bool('attention',False, 'Whether to use attention.')
flags.DEFINE_bool('bilinear',False, 'Whether to use bilinear decoder.')

flags.DEFINE_bool('multihead_attn',False,'Whether to use multi-headed attention.')
flags.DEFINE_integer('num_heads', 8, 'Number of attn heads.')
flags.DEFINE_integer('head_dim', 8, 'attn head dimention.')
flags.DEFINE_integer('num_heads_layer2', 4, 'Number of the attn heads in the second layer.')
flags.DEFINE_bool('average_attn',False,'Whether to use average as attn ageragation.')

flags.DEFINE_string('output_name','','Name of the output file to wirte results.')
flags.DEFINE_integer('num_experiments',1,'Number of experiments to run')
flags.DEFINE_bool('write_summary',False,'Whether to write summary.')
flags.DEFINE_bool('fixed_split',False,'Whether to randomly split dataset for each experiment.')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

l_roc = []
l_ap = []
l_acc = []
p=0.01
attrNoise = 0.2
m=1
if not FLAGS.fixed_split:
    for i in range(FLAGS.num_experiments):
        # Load data
        if dataset_str == 'synthetic':
            adj, features = get_synthetic_data(p=p, attrNoise=attrNoise)
        else:
            adj, features = load_data(dataset_str)
        #dense_feat = features.toarray()
        #sum_feat = np.sum(dense_feat,axis=1)
        #ave = np.sum(sum_feat)/len(sum_feat)
        #np.set_printoptions(threshold=sys.maxsize)
        #print(ave) #18
        #import pdb;pdb.set_trace()
        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj # sparse matrix
        # adj_orig.diagonal()[np.newaxis, :] row vector
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape) # set the diagnal elements to 0
        
        adj_orig.eliminate_zeros() # sparse matrix should not contain entries equals 0. So always call eliminate_zeros() after an update.

        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj,test_percent=10., val_percent=5.)
        adj = adj_train # This is the adj matrix that masked out all validation and testing entries.
        #print(adj_train.shape)
        #import pdb;pdb.set_trace()

        if FLAGS.features == 0:
            features = sp.identity(features.shape[0])  # featureless. sparse coo_matrix.

        # Some preprocessing
        #adj_norm = preprocess_graph(adj)

        if FLAGS.attention or FLAGS.multihead_attn:
            adj_norm = adj + sp.eye(adj.shape[0])
            adj_norm = sparse_to_tuple(adj_norm) # a tuple
        else:
            adj_norm = preprocess_graph(adj) # a tuple. Normalization. Identical matrix is added here.

        #print(type(adj + sp.eye(adj.shape[0])))
        #import pdb;pdb.set_trace()

        # Define placeholders 
        placeholders = { # this is passed directly to the model to build the graph.
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'in_drop': tf.placeholder_with_default(0., shape=()),
            'attn_drop': tf.placeholder_with_default(0., shape=()),
            'feat_drop': tf.placeholder_with_default(0., shape=())
        }

        num_nodes = adj.shape[0]

        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        sess = tf.Session()

        # Create model
        model = None
        if FLAGS.multihead_attn:
            model = MultiHeadedGAE(placeholders, num_features, features_nonzero)
        else:
            if model_str == 'gcn_ae':
                model = GCNModelAE(placeholders, num_features, features_nonzero, FLAGS.attention, FLAGS.bilinear)
            elif model_str == 'gcn_vae':
                model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, FLAGS.attention)

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        # Optimizer
        with tf.name_scope('optimizer'):
            if model_str == 'gcn_ae':
                opt = OptimizerAE(preds=model.reconstructions,
                                labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], # adj_orig in the original implementation
                                                                            validate_indices=False), [-1]),
                                pos_weight=pos_weight,
                                norm=norm)
            elif model_str == 'gcn_vae':
                opt = OptimizerVAE(preds=model.reconstructions,
                                labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], # adj_orig in the original implementation
                                                                            validate_indices=False), [-1]),
                                model=model, num_nodes=num_nodes,
                                pos_weight=pos_weight,
                                norm=norm)

        merged_summary = tf.summary.merge_all()
        # Initialize session
        with sess.as_default():
            sess.run(tf.global_variables_initializer())

            if FLAGS.write_summary:
                summary_writer = tf.summary.FileWriter('./summary/train/'+str(datetime.datetime.now()).replace(' ', '_'), sess.graph)

            cost_val = []
            acc_val = []


            def get_roc_score(edges_pos, edges_neg, emb=None):
                if emb is None:
                    feed_dict.update({placeholders['in_drop']: 0})
                    feed_dict.update({placeholders['attn_drop']: 0})
                    feed_dict.update({placeholders['feat_drop']: 0})
                    emb = sess.run(model.z_mean, feed_dict=feed_dict)

                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))

                # Predict on test set of edges
                adj_rec = np.dot(emb, emb.T)
                preds = []
                #pos = []
                for e in edges_pos:
                    preds.append(sigmoid(adj_rec[e[0], e[1]]))
                    #pos.append(adj_orig[e[0], e[1]])

                preds_neg = []
                #neg = []
                for e in edges_neg:
                    preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
                    #neg.append(adj_orig[e[0], e[1]])

                preds_all = np.hstack([preds, preds_neg]) #non-thresholded measure of decisions 
                labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
                roc_score = roc_auc_score(labels_all, preds_all)
                ap_score = average_precision_score(labels_all, preds_all)
                #print(preds_all)
                #import pdb;pdb.set_trace()
                #accuracy = accuracy_score(labels_all,preds_all)
                correct_prediction = np.equal(np.greater_equal(preds_all, 0.5).astype(np.int32),labels_all.astype(np.int32))
                accuracy = np.mean(correct_prediction.astype(np.float32))
                #print(accuracy.shape)
                #import pdb;pdb.set_trace()
                return roc_score, ap_score, accuracy


            cost_val = []
            acc_val = []
            val_roc_score = []

            adj_label = adj_train + sp.eye(adj_train.shape[0])
            adj_label = sparse_to_tuple(adj_label)

            epoch_time = 0
            # Train model
            for epoch in range(FLAGS.epochs):

                t = time.time()
                # Construct feed dictionary
                feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders) # global variable
                feed_dict.update({placeholders['in_drop']: FLAGS.in_drop}) # update is a methold of python dictionary
                feed_dict.update({placeholders['attn_drop']: FLAGS.attn_drop})
                feed_dict.update({placeholders['feat_drop']: FLAGS.feat_drop})
                # Run single weight update
                t0 = time.time()
                outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
                t1 = time.time()
                epoch_time += t1-t0
                # write summary
                if epoch % 5 == 0 and FLAGS.write_summary:
                    # Train set summary
                    summary = sess.run(merged_summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, epoch)
                    summary_writer.flush()

                # Compute average loss
                avg_cost = outs[1]
                avg_accuracy = outs[2]

                roc_curr, ap_curr,_ = get_roc_score(val_edges, val_edges_false)
                val_roc_score.append(roc_curr)
                
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                    "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
                    "val_ap=", "{:.5f}".format(ap_curr),
                    "time=", "{:.5f}".format(time.time() - t))
            end = time.time()
            print("sec/epoch: "+str(epoch_time/FLAGS.epochs))
            print("Optimization Finished!")
            
            # Test
            roc_score, ap_score, accuracy_score = get_roc_score(test_edges, test_edges_false)
            l_roc.append(roc_score)
            l_ap.append(ap_score)
            l_acc.append(accuracy_score)
            print('Test ROC score: ' + str(roc_score))
            print('Test AP score: ' + str(ap_score))
            print('Test Acc score: ' + str(accuracy_score))

            # get embeddings
            '''
            feed_dict.update({placeholders['in_drop']: 0})
            feed_dict.update({placeholders['attn_drop']: 0})
            feed_dict.update({placeholders['feat_drop']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)
            np.save('encoderout_nonparametric',emb)
            '''
        #attrNoise += 0.1
else:
    # Load data
    if dataset_str == 'synthetic':
        adj, features = get_synthetic_data()
    else:
        adj, features = load_data(dataset_str)
    #dense_feat = features.toarray()
    #sum_feat = np.sum(dense_feat,axis=1)
    #ave = np.sum(sum_feat)/len(sum_feat)
    #np.set_printoptions(threshold=sys.maxsize)
    #print(ave) #18
    #import pdb;pdb.set_trace()
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj # sparse matrix
    # adj_orig.diagonal()[np.newaxis, :] row vector
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape) # set the diagnal elements to 0
    
    adj_orig.eliminate_zeros() # sparse matrix should not contain entries equals 0. So always call eliminate_zeros() after an update.

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj,test_percent=10., val_percent=5.)
    adj = adj_train # This is the adj matrix that masked out all validation and testing entries.
    #print(adj_train.shape)
    #import pdb;pdb.set_trace()

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless. sparse coo_matrix.

    # Some preprocessing
    #adj_norm = preprocess_graph(adj)

    if FLAGS.attention or FLAGS.multihead_attn:
        adj_norm = adj + sp.eye(adj.shape[0])
        adj_norm = sparse_to_tuple(adj_norm) # a tuple
    else:
        adj_norm = preprocess_graph(adj) # a tuple. Normalization. Identical matrix is added here.

    #print(type(adj + sp.eye(adj.shape[0])))
    #import pdb;pdb.set_trace()

    # Define placeholders 
    placeholders = { # this is passed directly to the model to build the graph.
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'in_drop': tf.placeholder_with_default(0., shape=()),
        'attn_drop': tf.placeholder_with_default(0., shape=()),
        'feat_drop': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    
    for i in range(FLAGS.num_experiments):
        sess = tf.Session()

        # Create model
        model = None
        if FLAGS.multihead_attn:
            model = MultiHeadedGAE(placeholders, num_features, features_nonzero)
        else:
            if model_str == 'gcn_ae':
                model = GCNModelAE(placeholders, num_features, features_nonzero, FLAGS.attention, FLAGS.bilinear)
            elif model_str == 'gcn_vae':
                model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, FLAGS.attention)

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        # Optimizer
        with tf.name_scope('optimizer'):
            if model_str == 'gcn_ae':
                opt = OptimizerAE(preds=model.reconstructions,
                                labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], # adj_orig in the original implementation
                                                                            validate_indices=False), [-1]),
                                pos_weight=pos_weight,
                                norm=norm)
            elif model_str == 'gcn_vae':
                opt = OptimizerVAE(preds=model.reconstructions,
                                labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], # adj_orig in the original implementation
                                                                            validate_indices=False), [-1]),
                                model=model, num_nodes=num_nodes,
                                pos_weight=pos_weight,
                                norm=norm)

        merged_summary = tf.summary.merge_all()
        # Initialize session
        with sess.as_default():
            sess.run(tf.global_variables_initializer())

            if FLAGS.write_summary:
                summary_writer = tf.summary.FileWriter('./summary/train/'+str(datetime.datetime.now()).replace(' ', '_'), sess.graph)

            cost_val = []
            acc_val = []


            def get_roc_score(edges_pos, edges_neg, emb=None):
                if emb is None:
                    feed_dict.update({placeholders['in_drop']: 0})
                    feed_dict.update({placeholders['attn_drop']: 0})
                    feed_dict.update({placeholders['feat_drop']: 0})
                    emb = sess.run(model.z_mean, feed_dict=feed_dict)

                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))

                # Predict on test set of edges
                adj_rec = np.dot(emb, emb.T)
                preds = []
                #pos = []
                for e in edges_pos:
                    preds.append(sigmoid(adj_rec[e[0], e[1]]))
                    #pos.append(adj_orig[e[0], e[1]])

                preds_neg = []
                #neg = []
                for e in edges_neg:
                    preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
                    #neg.append(adj_orig[e[0], e[1]])

                preds_all = np.hstack([preds, preds_neg]) #non-thresholded measure of decisions 
                labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
                roc_score = roc_auc_score(labels_all, preds_all)
                ap_score = average_precision_score(labels_all, preds_all)
                #print(preds_all)
                #import pdb;pdb.set_trace()
                #accuracy = accuracy_score(labels_all,preds_all)
                correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_all), 0.5), tf.int32),
                                           tf.cast(labels_all, tf.int32))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                return roc_score, ap_score, accuracy


            cost_val = []
            acc_val = []
            val_roc_score = []

            adj_label = adj_train + sp.eye(adj_train.shape[0])
            adj_label = sparse_to_tuple(adj_label)

            epoch_time = 0
            # Train model
            for epoch in range(FLAGS.epochs):

                t = time.time()
                # Construct feed dictionary
                feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders) # global variable
                feed_dict.update({placeholders['in_drop']: FLAGS.in_drop}) # update is a methold of python dictionary
                feed_dict.update({placeholders['attn_drop']: FLAGS.attn_drop})
                feed_dict.update({placeholders['feat_drop']: FLAGS.feat_drop})
                # Run single weight update
                t0 = time.time()
                outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
                t1 = time.time()
                epoch_time += t1-t0
                # write summary
                if epoch % 5 == 0 and FLAGS.write_summary:
                    # Train set summary
                    summary = sess.run(merged_summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, epoch)
                    summary_writer.flush()

                # Compute average loss
                avg_cost = outs[1]
                avg_accuracy = outs[2]

                roc_curr, ap_curr,_ = get_roc_score(val_edges, val_edges_false)
                val_roc_score.append(roc_curr)
                
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                    "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
                    "val_ap=", "{:.5f}".format(ap_curr),
                    "time=", "{:.5f}".format(time.time() - t))
            end = time.time()
            print("sec/epoch: "+str(epoch_time/FLAGS.epochs))
            print("Optimization Finished!")
            
            # Test
            roc_score, ap_score, accuracy_score = get_roc_score(test_edges, test_edges_false)
            l_roc.append(roc_score)
            l_ap.append(ap_score)
            l_acc.append(accuracy_score)
            print('Test ROC score: ' + str(roc_score))
            print('Test AP score: ' + str(ap_score))
            print('Test Acc score: ' + str(accuracy_score))
            #import pdb;pdb.set_trace()
            # get embeddings
            '''
            feed_dict.update({placeholders['in_drop']: 0})
            feed_dict.update({placeholders['attn_drop']: 0})
            feed_dict.update({placeholders['feat_drop']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)
            np.save('encoderout_nonparametric',emb)
            '''

if FLAGS.num_experiments>1:
    l_roc = list(map(lambda x: x*100, l_roc))
    l_ap = list(map(lambda x: x*100, l_ap))
    with open('AGAE_attr.pkl', 'wb') as f:
        pickle.dump(l_acc,f)
        pickle.dump(l_roc,f)
        pickle.dump(l_ap,f)

    ave_roc = mean(l_roc)
    ave_ap = mean(l_ap)
    ave_acc = mean(l_acc)
    stdev_roc = pstdev(l_roc)
    stdev_ap = pstdev(l_ap)
    stdev_acc = np.std(l_acc)
    print('\nstdev test ROC = {}\nstdev test AP = {}'.format(str(stdev_roc),str(stdev_ap)))
    print('average test ROC = {}\naverage test AP = {}'.format(str(ave_roc),str(ave_ap)))
    print('average test Acc = {}\nstdev test Acc = {}'.format(str(ave_acc),str(stdev_acc)))
if FLAGS.output_name != '':
    with open(FLAGS.output_name,'a') as f:
        f.write('average test ROC = {}\naverage test AP = {}\n'.format(str(ave_roc),str(ave_ap)))
        if FLAGS.num_experiments>1:
            f.write('stdev test ROC = {}\nstdev test AP = {}\n\n'.format(str(stdev_roc),str(stdev_ap)))
            f.write('average test Acc = {}\nstdev test Acc = {}'.format(str(ave_acc),str(stdev_acc)))