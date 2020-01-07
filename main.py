import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
import numpy as np
from my_generator import Vgg19 as Vgg
import pdb
import batch_data.image_data as dataset
import scipy.io as sio
import matplotlib.pyplot as plt

def calc_map_k(qB, rB, query_L, retrieval_L,k):
    map = 0
    gnd2 = (np.dot(query_L, retrieval_L.transpose()) > 0).astype(np.float32)
    gnd2 = gnd2.squeeze()
    hamm = calc_hammingDist(qB, rB)
    aaa = np.arange(0,retrieval_L.shape[0])
    ind = np.lexsort((aaa,hamm))
    #pdb.set_trace()
    gnd = gnd2[ind[:k]]
    tsum = np.sum(gnd)
    if tsum == 0:
        map = 0
    else:
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    return map

def calc_hammingDist(te_code, db_code):
    ind = te_code < 0.5
    te_code[ind] = -1
    ind = db_code < 0.5
    db_code[ind] = -1
    q = db_code.shape[1]
    distH = q - np.dot(te_code, db_code.transpose())
    return distH

#show retrieval results.
def show_retrieval_results(te_img, te_code,te_label, db_code, db_data, name):
    hamm = calc_hammingDist(te_code, db_code)
    aaa = np.arange(0, db_code.shape[0])
    #pdb.set_trace()
    ind = np.lexsort((aaa, hamm))
    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 12), (0,0))
    #te_img = te_img[0]
    ax1.imshow(te_img)
    ax2 = plt.subplot2grid((2, 12), (1,0))
    ax2.text(1,1,str(te_label),fontsize = 1)
    for i in range(10):
        temp_ax = plt.subplot2grid((2, 12), (0,i+2))
        temp_db_img, temp_db_label = db_data.img_data([ind[i]])
        temp_db_img = temp_db_img[0]
        temp_ax.imshow(temp_db_img)
        temp_ax2 = plt.subplot2grid((2, 12), (1,i+2))
        temp_ax2.text(1,1,str(temp_db_label), fontsize = 1)
    fig.savefig(name)
    plt.close(fig)

def adversarial_loss(ad_code, or_code, threshold_t, beta):
    ad_code, or_code,s = mask_code(ad_code, or_code, threshold_t)
    #s = tf.sign(tf.abs(ad_code- tf.sign(or_code))-(1+threshold))
    m = tf.reduce_sum(s)
    ad_loss = tf.reduce_mean(tf.square(tf.multiply(1.0/m, tf.matmul(or_code, tf.transpose(ad_code))) +1)) + tf.multiply(beta, tf.square(tf.reduce_mean(ad_code)))
    return ad_loss,s

def mask_code(ad_code, or_code,threshold_t):
    s = tf.sign((1+threshold_t) - tf.abs(ad_code- tf.sign(or_code))) 
    s = tf.multiply(1.0/2, s+1)
    ad_code = tf.multiply(s, ad_code)
    or_code = tf.multiply(s, or_code)
    return ad_code, or_code, s
    

# dataset_config
config = {
    
    'img_tr': "cifar10/img_train.txt", 
    'lab_tr': "cifar10/label_train.txt",
    'img_te': "cifar10/img_test.txt",
    'lab_te': "cifar10/label_test.txt",
    'img_db': "cifar10/img_database.txt", 
    'lab_db': "cifar10/label_database.txt",
    'n_train': 5000,
    'n_test': 10000,
    'n_db': 50000,
    'n_label': 10
}
test_data = dataset.import_test(config)
db_data = dataset.import_db(config)
#pdb.set_trace()
#pdb.set_trace()
threshold = 0.5

# Initial net
sess = tf.Session()
hidden_size = 16 
x224 = tf.placeholder(tf.float32, [None, 224,224,3])
x_hat = tf.Variable(tf.zeros( [1,224,224,3]))
original_code = tf.Variable(tf.zeros([hidden_size]), trainable = False) 
lr_E = tf.placeholder(tf.float32, shape=[])
alpha = tf.placeholder(tf.float32, shape=[])
learning_rate = 100
beta = 1.0

train_mode = tf.placeholder(tf.bool)
net = Vgg('./My-save-cifar10-16.npy', codelen = hidden_size)
net.build(x_hat, train_mode)
pre_z_x = net.fc8
z_x = tf.nn.tanh(tf.multiply(alpha, pre_z_x))

sess.run(tf.initialize_all_variables())

# code for database
temp_db_data = np.load('cifar10-16bit-99.npy')
temp_db_code = temp_db_data.item()['dataset_codes']
temp_db_label = temp_db_data.item()['dataset_L']

#project
epsilon = tf.placeholder(tf.float32, ())
below = x224 - epsilon
above = x224 + epsilon
projected = tf.clip_by_value(x_hat, below, above)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)
#Optimizer
ad_loss, thre_s = adversarial_loss(z_x, original_code, threshold, beta)
opt_ad = tf.train.AdamOptimizer(lr_E)
opt_ad = tf.train.GradientDescentOptimizer(lr_E)
train_op = opt_ad.minimize(ad_loss, var_list = [x_hat])

#Iteration for test images
demo_epsilon = 10.0
epoch = 2000
test_num = 200
move_map = 0
move_ad_map = 0
for test_ind in range(test_num):
    #Code for original image
    #test_ind = 2
    test_image, test_label = test_data.img_data([test_ind])
    assign_op = tf.assign(x_hat, x224)
    sess.run(assign_op, feed_dict = {x224: test_image})
    temp_code = sess.run(z_x, feed_dict={alpha: 1, train_mode: False})
    temp_code = np.squeeze(temp_code)
    assign_code = tf.assign(original_code, temp_code)
    sess.run(assign_code)
    temp_code2 = (temp_code>0)*2-1
    #show_retrieval_results(test_image[0], temp_code, test_label,temp_db_code, db_data, 'original.pdf')
    test_map = calc_map_k(temp_code2, temp_db_code, test_label, temp_db_label, 1000)
    move_map = (move_map*(test_ind) + test_map)/(test_ind + 1)


    #training
    n = 0
    epoch2test = 10
    total_num = epoch/epoch2test

    all_ad_img = np.zeros([total_num, 224,224,3], dtype = np.uint8)
    all_ad_code = np.zeros([total_num, hidden_size])
    all_Hamming_dist = np.zeros(total_num)
    all_mask = np.zeros([total_num, hidden_size])
    all_ad_map = np.zeros(total_num)
    all_loss = np.zeros(total_num)
    alpha_vector = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7,1.0]
    alpha_value = 0.1
    epoch2alpha = epoch/10

    while n < epoch:
        if n%epoch2alpha == 0:
            alpha_value = alpha_vector[n/epoch2alpha]
        if n%epoch2test == 0:
            temp_loss,temp_hat_code, mask_s = sess.run([ad_loss,z_x, thre_s], feed_dict={alpha: alpha_value, lr_E: learning_rate, train_mode: False})
            temp_hat_code2 = (temp_hat_code>0)*2-1
            Hamming_dist = hidden_size - np.dot((temp_code2), np.transpose(temp_hat_code2))
            if n%20 == 0:
                print "test_ind:{} epoch: {}, loss: {}, Hamming distance: {},threshold:{},temp_code:{},temp_hat_code:{}".format(test_ind, n, temp_loss, Hamming_dist, mask_s, temp_code, temp_hat_code)
            temp_hat_code2 = temp_hat_code2.squeeze()
            ad_img = sess.run(x_hat)
            ad_img = ad_img.squeeze()
            ad_img = ad_img.astype(np.uint8)
            temp_num = n/epoch2test
            all_ad_img[temp_num,:,:,:] = ad_img
            all_ad_code[temp_num, :] = temp_hat_code2
            all_Hamming_dist[temp_num] = Hamming_dist
            all_mask[temp_num,:] = mask_s
            all_ad_map[temp_num] = calc_map_k(temp_hat_code2, temp_db_code, test_label, temp_db_label, 1000)
            all_loss[temp_num] = temp_loss
            if Hamming_dist == 2*hidden_size:
                break

        sess.run([train_op], feed_dict={alpha: alpha_value, lr_E: learning_rate, train_mode: True})
        sess.run(project_step, feed_dict = {x224: test_image, epsilon: demo_epsilon})
        n = n+1

    move_ad_map = (move_ad_map*(test_ind) + all_ad_map[total_num-1])/(test_ind + 1)
    print 'move_map:{}, move_ad_map:{}'.format(move_map, move_ad_map)
    save_name = './result/cifar10/16bit/threshold_0.5_map_k_ResultOfTestImage_' + str(test_ind) + '.npy'
    dict_ = {'test_image': test_image[0], 'test_code': temp_code2, 'ad_image': ad_img, 'ad_code': temp_hat_code2, 'all_ad_code': all_ad_code, 'all_Hamming_dist': all_Hamming_dist, 'all_mask':all_mask,'all_ad_map': all_ad_map, 'test_map': test_map, 'move_map': move_map, 'move_ad_map': move_ad_map}
    np.save(save_name, dict_)
            
        
        
        
    
    
    
        
    
