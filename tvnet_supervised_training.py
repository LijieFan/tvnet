import numpy as np
import cv2
import os
import scipy.io as sio
from tvl1_flow_trainable import Tvl1Flow
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("scale", 1, " TVNet scale [3]")
flags.DEFINE_integer("warp", 1, " TVNet warp [1]")
flags.DEFINE_integer("iteration", 50, " TVNet iteration [10]")
flags.DEFINE_string("gpu", '0', " gpu to use [0]")
FLAGS = flags.FLAGS

scale = FLAGS.scale
warp = FLAGS.warp
iteration = FLAGS.iteration
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

print 'TVNet Testing Params:\n scale: %d\n warp: %d\n iteration: %d\n gpu: %s' \
      % (scale, warp, iteration, FLAGS.gpu)

w = h = 240
# describe graph
batch_size = 32
x1 = tf.placeholder(shape=[batch_size, h, w, 3], dtype=tf.float32)
x2 = tf.placeholder(shape=[batch_size, h, w, 3], dtype=tf.float32)
gt = tf.placeholder(shape=[batch_size, h, w, 2], dtype=tf.float32)
flow = Tvl1Flow()
unsupervised_loss, u1, u2 = flow.get_loss(x1,x2,max_scales=scale,
                     warps=warp,
                     max_iterations=iteration)
u1_gt_tf = tf.reshape(gt[:,:,:,0],[batch_size,h,w,1])
u2_gt_tf = tf.reshape(gt[:,:,:,1],[batch_size,h,w,1])
supervised_loss = flow.supervised_loss(u1, u2, u1_gt_tf, u2_gt_tf)

gvs = tf.trainable_variables()
Inception_VarList = [i for i in gvs]

supervised_train_op = tf.train.AdamOptimizer(0.005).minimize(supervised_loss, var_list=Inception_VarList)
unsupervised_train_op = tf.train.AdamOptimizer(0.005).minimize(unsupervised_loss, var_list=Inception_VarList)

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))
sess.run(tf.global_variables_initializer())


choose_able_index = []
Dataset_base_path = 'MiddleBury/data'
GT_base_path = 'MiddleBury/flow'
frame1_name = 'frame10.png'
frame2_name = 'frame11.png'
flow_name = 'flow10.flo'

snap_dir = 'snapshots'
if not os.path.exists(snap_dir):
    os.mkdir(snap_dir)

category_list = os.listdir(Dataset_base_path)
category_list.sort()
for category in category_list:
    category_path = os.path.join(Dataset_base_path, category)
    if not os.path.isdir(category_path):
        category_list.remove(category)

image_num = len(category_list)
image1_matrix = np.zeros([image_num, w, h, 3])
image2_matrix = np.zeros([image_num, w, h, 3])
gt_matrix = np.zeros([image_num, w, h, 2])
UNKNOWN_FLOW_THRESH = 1e9
for cat_idx, category in enumerate(category_list):
    print category
    # read in image
    category_path = os.path.join(Dataset_base_path, category)
    image1_path = os.path.join(category_path, frame1_name)
    image2_path = os.path.join(category_path, frame2_name)
    img1 = cv2.imread(image1_path)
    h_gt, w_gt, channel = img1.shape
    img2 = cv2.imread(image2_path)
    img1 = cv2.resize(img1, (w,h))
    img2 = cv2.resize(img2, (w,h))
    image1_matrix[cat_idx,:,:,:] = img1
    image2_matrix[cat_idx,:,:,:] = img2
    # read in gt flow
    gt_full_path = os.path.join(GT_base_path, category, flow_name)
    fid_gt = open(gt_full_path, 'rb')
    data_gt = np.fromfile(fid_gt, dtype=np.float32)
    data_gt = data_gt[3:].reshape((h_gt, w_gt, 2))
    u1_gt = data_gt[:, :, 0]
    u2_gt = data_gt[:, :, 1]
    u1_gt[np.abs(u1_gt)>UNKNOWN_FLOW_THRESH] = 0
    u2_gt[np.abs(u2_gt)>UNKNOWN_FLOW_THRESH] = 0
    u1_gt = cv2.resize(u1_gt, (w, h))
    u1_gt /= (w_gt+0.) / w
    u2_gt= cv2.resize(u2_gt, (w, h))
    u2_gt /= (h_gt+0.) / h
    # delete unknown val
    gt_matrix[cat_idx,:,:,0] = u1_gt
    gt_matrix[cat_idx,:,:,1] = u2_gt

display_iter = 1.
loss_saver = 0
u_loss_saver = 0
total_iter = 2000
save_iter = 1000
saver = tf.train.Saver([var for var in tf.global_variables()],max_to_keep=2)
for i in xrange(total_iter):
    x1_np = np.zeros([batch_size,h,w,3])
    x2_np = np.zeros([batch_size,h,w,3])
    gt_np = np.zeros([batch_size,h,w,2])
    choosen_idx = np.random.random_integers(0,image_num-1,[batch_size,])
    for ii in xrange(batch_size):
        x1_np[ii,:,:,:] = image1_matrix[choosen_idx[ii],:,:,:]
        x2_np[ii,:,:,:] = image2_matrix[choosen_idx[ii],:,:,:]
        gt_np[ii,:,:,:] = gt_matrix[choosen_idx[ii],:,:,:]
    _,cur_loss,cur_u_loss = sess.run([supervised_train_op, supervised_loss, unsupervised_loss], feed_dict={x1: x1_np, x2: x2_np, gt: gt_np})
    loss_saver += cur_loss / display_iter
    u_loss_saver += cur_u_loss / display_iter
    if (i+1)%display_iter == 0:
        print 'training iter: %d, avg loss: %f, avg aee: %f'%((i+1), u_loss_saver, loss_saver)
        loss_saver = 0
        u_loss_saver = 0

    if (i+1)%save_iter == 0:
        save_file_name = os.path.join(snap_dir,'supervised_%d_%d_%d_iter_%d.ckpt'%(scale, warp, iteration, i+1))
        print 'saving iter: %d, save to file %s'%(i+1, save_file_name)
        saver.save(sess,save_file_name)

