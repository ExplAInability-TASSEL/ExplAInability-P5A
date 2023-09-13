import sys
import os
import numpy as np
import math
import random
from operator import itemgetter, attrgetter, methodcaller
import tensorflow as tf
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.rnn import DropoutWrapper
import time
from tensorflow.contrib import rnn
from GUnit import GUnit
from sklearn.cluster import KMeans
import pickle
#from tensorflow.contrib.rnn import DropoutWrapper


#OK
def customLogLoss(pred, y):
	nrow, ncol = pred.shape
	tot = 0
	for i in range(nrow):
		for j in range(ncol):
			if pred[i,j] == 0:
				tot +=0
			else:
				tot+= -1 * y[i,j] * np.log(pred[i,j])
	return tot / nrow





#OK
def checkTest(ts_data_S2, batchsz, label_test, toPrint, data_limit, bins, nclasses):
	tot_pred = []
	tot_sm = None
	#tot_pred2 = []
	iterations = ts_data_S2.shape[0] / batchsz

	if ts_data_S2.shape[0] % batchsz != 0:
	    iterations+=1

	# sigma_val = None
	for ibatch in range(iterations):
		batch_rnn_x_S2, batch_limit = getBatch(ts_data_S2, data_limit, ibatch, batchsz)
		batch_mask = np.zeros((batch_limit.shape[0],bins))
		for idx, val in enumerate(batch_limit):
			for i in range(val):
				batch_mask[idx,i] = 1.0
		pred_temp, pred_sm = sess.run([testPrediction,pred_tot],feed_dict={
													   x_data:batch_rnn_x_S2,
													   dropOut:0.,
													   is_training_ph:False,
													   seq_length:batch_limit,
													   #batch_size:len(batch_limit),
													   mask:batch_mask
														})

		# del batch_rnn_x_S1
		del batch_rnn_x_S2
		#del batch_y
		if tot_sm is None:
			tot_sm = pred_sm
		else:
			tot_sm = np.concatenate((tot_sm,pred_sm),axis=0)

		for el in pred_temp:
			tot_pred.append( el )

	#print (np.bincount(np.array(tot_pred)))
	#print (np.bincount(np.array(label_test)))
	if toPrint:
		print ("PREDICTION")
		print ("TEST F-Measure: %f" % f1_score(label_test, tot_pred, average='weighted'))
		print (f1_score(label_test, tot_pred, average=None))
		print ("TEST Accuracy: %f" % accuracy_score(label_test, tot_pred))
		print ("LOG LOSS: %f" % log_loss( tf.keras.utils.to_categorical(label_test, num_classes=nclasses), tot_sm ))
		print("CUSTOM LOG LOSS %f" % customLogLoss( tot_sm, tf.keras.utils.to_categorical(label_test, num_classes=nclasses)))
		#print(confusion_matrix(label_test, tot_pred))

	sys.stdout.flush()
	#return f1_score(label_test,tot_pred,average="weighted")#accuracy_score(label_test, tot_pred)
	#return log_loss( tf.keras.utils.to_categorical(label_test, num_classes=7), tot_sm )
	#return customLogLoss(tot_sm, tf.keras.utils.to_categorical(label_test, num_classes=7))
	return f1_score(label_test, tot_pred, average='weighted')




#OK
def getBatch(X, Y, i, batch_size):
    start_id = i*batch_size
    end_id = min( (i+1) * batch_size, X.shape[0])
    batch_x = X[start_id:end_id]
    batch_y = Y[start_id:end_id]
    return batch_x, batch_y

##########################################################################################################################################################









##########@XGBBOOST###########
#OK
def getSL(data):
	sl = []
	for el in data:
		s = np.sum(el,axis=1)
		idx = np.where(s == 0)
		idx = idx[0]
		if len(idx) == 0:
			sl.append(len(s))
		else:
			sl.append(idx[0])
	return np.array(sl)


splits = sys.argv[1]
nclust = sys.argv[2]
prefix = sys.argv[3]
train_data = np.load(prefix+"/splits/train_"+str(splits)+"_BoP_"+nclust+".npy")
valid_data = np.load(prefix+"/splits/valid_"+str(splits)+"_BoP_"+nclust+".npy")
print("train_Data ",train_data.shape)
print("valid_data ",valid_data.shape)
train_data = train_data[:,:,0:(21*6)]
valid_data = valid_data[:,:,0:(21*6)]



train_label = np.load(prefix+"/splits/train_"+str(splits)+"_labels.npy")
valid_label = np.load(prefix+"/splits/valid_"+str(splits)+"_labels.npy")
train_sl = getSL(train_data)
valid_sl = getSL(valid_data)

'''
print("train_label ",train_label.shape)
print("valid_label ",valid_label.shape)
print("train_data ",train_data.shape)
print("valid_data ",valid_data.shape)
'''
#exit()

#print("np.amax ",np.amax(train_sl))
#print("np.amin ",np.amin(train_sl))

#print( np.bincount(train_sl) )

#exit()

#train_sl = np.load("splits/train_"+str(splits)+"_sl.npy")
#valid_sl = np.load("splits/valid_"+str(splits)+"_sl.npy")


print("train_data ", train_data.shape)
print("train_data ", train_data.shape)


nfeat = train_data.shape[2]
bins = train_data.shape[1]
nclasses = len(np.unique(train_label))
print("nfeat ",nfeat)
print("nclasses ", nclasses)

train_y = tf.keras.utils.to_categorical(train_label)


########### TRANSF ###########
tf.reset_default_graph()
# Add ops to save and restore all the variables.
ckpt_path = "REUNION_data_results/BoP_CNN_SM_2/model_2"
sess = tf.InteractiveSession()
#with tf.Session() as sess:
model_saver = tf.train.import_meta_graph(ckpt_path+".meta")
model_saver.restore(sess, ckpt_path)
graph = tf.get_default_graph()

x_data = graph.get_tensor_by_name("x_data:0")
mask = graph.get_tensor_by_name("mask:0")
seq_length = graph.get_tensor_by_name("limits:0")
is_training_ph = graph.get_tensor_by_name("is_training:0")
dropOut = graph.get_tensor_by_name("drop_rate:0")
features = graph.get_tensor_by_name("features:0")
y = tf.placeholder("float",[None,nclasses],name="y_new")

aux_cl = tf.layers.dense(features, nclasses, activation=None)


feat = tf.keras.layers.Dense(256,activation='relu')(features)
feat = tf.layers.batch_normalization(feat)

feat = tf.keras.layers.Dense(256,activation='relu')(feat)
feat = tf.layers.batch_normalization(feat)

#feat = tf.keras.layers.Dense(128,activation='relu')(feat)
#feat = tf.layers.batch_normalization(feat)

temp_pred = tf.layers.dense(feat, nclasses, activation=None)

with tf.variable_scope("pred_env_new"):
	pred_tot = tf.nn.softmax( temp_pred )
	testPrediction = tf.argmax(pred_tot, 1, name="prediction")
	correct = tf.equal(tf.argmax(pred_tot,1),tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct,tf.float64))

with tf.variable_scope("cost_new"):
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=temp_pred)
	loss_aux = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=aux_cl)
	cost = tf.reduce_mean(loss) + .5*tf.reduce_mean(loss_aux)


	#reg = tf.reduce_mean( tf.reduce_sum(alphas_b,axis=1) )
	#reg = tf.zeros(1)
	#cost = cost #+ (.01 * reg )

train_op = tf.train.AdamOptimizer(learning_rate=0.0001,name="new_opt").minimize(cost)

global_vars= tf.global_variables()
is_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
not_initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if not f]
print "GLOBAL VARS: %d, NOT INITIALIZED: %d" % ( len(global_vars), len(not_initialized_vars) )

#if len(not_initialized_vars):
sess.run(tf.variables_initializer(not_initialized_vars))


###################### END OF COMPUTATIONAL GRAPH ################################""
def randomHorizontalShift(train_data, train_data_limit, max_len):
	new_data = []
	for idx, row in enumerate(train_data):
		_, ndim = row.shape
		limit = train_data_limit[idx]
		temp = row[0:limit,:]
		temp = shuffle(temp)
		toappend = np.zeros((max_len-limit,ndim))
		temp = np.concatenate((temp,toappend ),axis=0)
		new_data.append(temp)
	return np.array(new_data)


#tf.global_variables_initializer().run()
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

batchsz = 32
hm_epochs = 5000
output_dir_models = prefix+"/BoP_TRANSF_SM_"+nclust

iterations = train_data.shape[0] / batchsz

if train_data.shape[0] % batchsz != 0:
    iterations+=1

best_f1 = sys.float_info.min

print ("n iterations: %d" % iterations)

for e in range(hm_epochs):
	start = time.time()
	lossi = 0
	accS = 0
	tot_alphas = None
	train_data,  train_sl, train_y = shuffle(train_data, train_sl, train_y)
	train_data = randomHorizontalShift(train_data, train_sl, bins)

	for ibatch in range(iterations):
		batch_limit, _ = getBatch(train_sl, train_sl, ibatch, batchsz)
		batch_mask = np.zeros((batch_limit.shape[0],bins))
		for idx, val in enumerate(batch_limit):
			batch_mask[idx,0:val] = 1.0
		batch_x, batch_y = getBatch(train_data, train_y, ibatch, batchsz)
		acc,_,loss, t_pred = sess.run([accuracy, train_op ,cost, testPrediction],feed_dict={ x_data:batch_x,
														y:batch_y,
														dropOut:0.4,
														is_training_ph:True,
														seq_length:batch_limit,
														mask:batch_mask
														})
		lossi+=loss
		accS+=acc

		del batch_x
		del batch_y

		done = time.time()
		elapsed = done - start
	print ("Epoch: ",e," Train loss:",lossi/iterations," | accuracy:",accS/iterations, " | time: ",elapsed	)
	print("valid_data ",valid_data.shape)
	print("valid_label ",valid_label.shape)
	print("valid_sl ",valid_sl.shape)

	c_loss = lossi/iterations
	val_f1 = checkTest(valid_data, 1024, valid_label, False, valid_sl, bins, nclasses)

	print(" val_f1 >  best_f1 ", val_f1, " > ",best_f1)
	if val_f1 > best_f1:
		save_path = saver.save(sess, output_dir_models+"/model_"+str(splits))
		checkTest(valid_data, 1024, valid_label, True, valid_sl,bins, nclasses)
		print("Model saved in path: %s" % save_path)
		best_f1 = val_f1
