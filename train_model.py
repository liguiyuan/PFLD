# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))

import tensorflow as tf
import numpy as np
import cv2
import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import time

from generate_data import DateSet
from model2 import create_model
from utils import train_model
from euler_angles_utils import calculate_pitch_yaw_roll

def main(args):
    debug = (args.debug == 'True')
    print(args)
    np.random.seed(args.seed)
    with tf.Graph().as_default():
        train_dataset, num_train_file = DateSet(args.file_list, args, debug)
        test_dataset, num_test_file = DateSet(args.test_list, args, debug)
        list_ops = {}

        batch_train_dataset = train_dataset.batch(args.batch_size).repeat()
        train_iterator = batch_train_dataset.make_one_shot_iterator()
        train_next_element = train_iterator.get_next()

        batch_test_dataset = test_dataset.batch(args.batch_size).repeat()
        test_iterator = batch_test_dataset.make_one_shot_iterator()
        test_next_element = test_iterator.get_next()

        list_ops['num_train_file'] = num_train_file
        list_ops['num_test_file'] = num_test_file

        model_dir = args.model_dir
        if 'test' in model_dir and debug and os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
        #assert not os.path.exists(model_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        print('Total number of examples: {}'.format(num_train_file))
        print('Test number of examples: {}'.format(num_test_file))
        print('Model dir: {}'.format(model_dir))

        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        list_ops['global_step'] = global_step
        list_ops['train_dataset'] = train_dataset
        list_ops['test_dataset'] = test_dataset
        list_ops['train_next_element'] = train_next_element
        list_ops['test_next_element'] = test_next_element

        epoch_size = num_train_file//args.batch_size
        print('Number of batches per epoch: {}'.format(epoch_size))

        image_batch = tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3),\
                                     name='image_batch')
        landmark_batch = tf.placeholder(tf.float32, shape=(None, 196), name='landmark_batch')
        attribute_batch = tf.placeholder(tf.int32,  shape=(None, 6), name='attribute_batch')
        euler_angles_gt_batch = tf.placeholder(tf.float32,  shape=(None, 3), name='euler_angles_gt_batch')
        w_n = tf.placeholder(tf.float32,shape=(None),name='w_n')
        
        list_ops['image_batch'] = image_batch
        list_ops['landmark_batch'] = landmark_batch
        list_ops['attribute_batch'] = attribute_batch
        list_ops['euler_angles_gt_batch'] = euler_angles_gt_batch
        list_ops['w_n'] = w_n

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        list_ops['phase_train_placeholder'] = phase_train_placeholder

        print('Building training graph.')
        # total_loss, landmarks, heatmaps_loss, heatmaps= create_model(image_batch, landmark_batch,\
        #                                                                                phase_train_placeholder, args)

        landmarks_pre, landmarks_loss,euler_angles_pre = create_model(image_batch, landmark_batch,\
                                                                              phase_train_placeholder, args)

        L2_loss = tf.add_n(tf.losses.get_regularization_losses())
        _sum_k = tf.reduce_sum(tf.map_fn(lambda x: 1 - tf.cos(abs(x)), \
                                         euler_angles_gt_batch - euler_angles_pre), axis=1)

        loss_sum = tf.reduce_sum(tf.square(landmark_batch - landmarks_pre), axis=1)
        loss_sum = tf.reduce_mean(loss_sum*_sum_k*w_n)
        loss_sum += L2_loss

        train_op, lr_op = train_model(loss_sum, global_step, num_train_file, args)

        list_ops['landmarks'] = landmarks_pre
        list_ops['L2_loss'] = L2_loss
        list_ops['loss'] = loss_sum
        list_ops['train_op'] = train_op
        list_ops['lr_op'] = lr_op

        save_params = tf.trainable_variables()
        saver = tf.train.Saver(save_params, max_to_keep=None)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=False,log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        with sess.as_default():
            if args.pretrained_model:
                pretrained_model = args.pretrained_model
                if (not os.path.isdir(pretrained_model)):
                    print('Restoring pretrained model: {}'.format(pretrained_model))
                    saver.restore(sess, args.pretrained_model)
                else:
                    print('Model directory: {}'.format(pretrained_model))
                    ckpt = tf.train.get_checkpoint_state(pretrained_model)
                    model_path = ckpt.model_checkpoint_path
                    assert (ckpt and model_path)
                    print('Checkpoint file: {}'.format(model_path))
                    saver.restore(sess, model_path)

            if args.save_image_example:
                save_image_example(sess, list_ops, args)

            print('Running train.')
            for epoch in range(args.max_epoch):
                train(sess, epoch_size, epoch, list_ops)
                checkpoint_path = os.path.join(model_dir,'model.ckpt')
                metagraph_path = os.path.join(model_dir,'model.meta')
                print('Saving variables')
                start_time = time.time()
                saver.save(sess, checkpoint_path, global_step=epoch,write_meta_graph=False)
                save_time_variables = time.time() - start_time
                print('Variables saved in %.2f seconds' % save_time_variables)

                if not os.path.exists(metagraph_path):
                    saver.export_meta_graph(metagraph_path)

                test(sess, list_ops, args)

def train(sess, epoch_size, epoch, list_ops):

    image_batch, landmarks_batch, attribute_batch = list_ops['train_next_element']

    TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    for i in range(epoch_size):

        # # print(images.shape, landmarks.shape, attributes.shape)
        #TODO : get the w_n and euler_angles_gt_batch
        images, landmarks, attributes = sess.run([image_batch, landmarks_batch, attribute_batch])
        euler_angles_landmarks = []
        for index in TRACKED_POINTS:
            euler_angles_landmarks.append(landmarks[:,2*index:2*index+2])
        euler_angles_landmarks = np.asarray(euler_angles_landmarks).reshape((-1,28))
        euler_angles_gt = []
        for j in range(euler_angles_landmarks.shape[0]):
            pitch , yaw ,roll = calculate_pitch_yaw_roll(euler_angles_landmarks[j])
            euler_angles_gt.append((pitch,yaw,roll))
        euler_angles_gt = np.asarray(euler_angles_gt).reshape((-1,3))

        '''
        calculate the w_n: return the batch [-1,1]
        c :
        #201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        #202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        #203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        #204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        #205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        '''
        attributes_w_n= tf.to_float(attributes[:,1:6])
        _num = attributes_w_n.shape[0]
        mat_ratio = tf.reduce_mean(attributes_w_n,axis=0)
        #TODO when use function tf.map_fn get error results [inf,nan]
        # mat_ratio = tf.map_fn(lambda x:1.0/x if not x==0.0 else 0.0,mat_ratio)
        #mat_ratio = map(lambda x: 1.0 / x if not x == 0.0 else float(images.shape[0]), sess.run(mat_ratio))
        mat_ratio = list(map(lambda x: 1.0 / x if not x == 0.0 else float(images.shape[0]), sess.run(mat_ratio)))
        attributes_w_n = attributes_w_n*mat_ratio
        attributes_w_n = tf.reduce_sum(attributes_w_n,axis=1)
        # attributes_w_n = tf.expand_dims(attributes_w_n,1)
        # attributes_w_n = tf.tile(attributes_w_n,[_num])
        #TODO change the value of the zero in mat
        attributes_w_n = sess.run(attributes_w_n)

        feed_dict = {
            list_ops['image_batch']: images,
            list_ops['landmark_batch']: landmarks,
            list_ops['attribute_batch']: attributes,
            list_ops['phase_train_placeholder']: True,
            list_ops['euler_angles_gt_batch'] : euler_angles_gt,
            list_ops['w_n']: attributes_w_n
        }
        loss, _, lr, L2_loss = sess.run([list_ops['loss'], list_ops['train_op'], list_ops['lr_op'],\
                        list_ops['L2_loss']], feed_dict=feed_dict)

        if ((i + 1) % 10) == 0 or (i+1) == epoch_size:
            Epoch = 'Epoch:[{:<4}][{:<4}/{:<4}]'.format(epoch, i+1, epoch_size)
            Loss = 'Loss {:2.3f}\tL2_loss {:2.3f}'.format(loss, L2_loss)
            print('{}\t{}\t lr {:2.3}'.format(Epoch, Loss, lr))

def test(sess, list_ops, args):
    image_batch, landmarks_batch, attribute_batch = list_ops['test_next_element']

    sample_path = os.path.join(args.model_dir, 'HeatMaps')
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    loss_sum = 0
    landmark_error = 0
    landmark_01_num=0

    epoch_size = list_ops['num_test_file'] // args.batch_size
    for i in range(epoch_size): #batch_num
        images, landmarks, attributes = sess.run([image_batch, landmarks_batch, attribute_batch])
        feed_dict = {
            list_ops['image_batch']: images,
            list_ops['landmark_batch']: landmarks,
            list_ops['attribute_batch']: attributes,
            list_ops['phase_train_placeholder']: False
        }
        pre_landmarks = sess.run(list_ops['landmarks'], feed_dict=feed_dict)

        diff = pre_landmarks - landmarks
        loss = np.sum(diff * diff)
        loss_sum += loss

        for k in range(pre_landmarks.shape[0]):
            error_all_points=0
            for count_point in range(pre_landmarks.shape[1]//2): #num points
                error_diff=pre_landmarks[k][(count_point*2):(count_point*2+2)]-landmarks[k][(count_point*2):(count_point*2+2)]
                error = np.sqrt(np.sum(error_diff * error_diff))
                error_all_points += error
            interocular_distance=np.sqrt(np.sum(pow((landmarks[k][120:122]-landmarks[k][144:146]),2)))
            error_norm=error_all_points/(interocular_distance*98)
            landmark_error += error_norm
            if error_norm >=0.1 :
                landmark_01_num += 1

        if i == 0:
            image_save_path = os.path.join(sample_path, 'img')
            if not os.path.exists(image_save_path):
                os.mkdir(image_save_path)

            for j in range(images.shape[0]): #batch_size
                image = images[j]*256
                image = image[:,:,::-1]

                image_i = image.copy()
                pre_landmark = pre_landmarks[j]
                h, w, _ = image_i.shape
                pre_landmark = pre_landmark.reshape(-1, 2) * [w, h]
                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(image_i, (x, y), 1, (0, 0, 255))
                landmark = landmarks[j].reshape(-1, 2) * [w, h]
                for (x, y) in landmark.astype(np.int32):
                    cv2.circle(image_i, (x, y), 1, (255, 0, 0))
                image_save_name = os.path.join(image_save_path, '{}.jpg'.format(j))
                cv2.imwrite(image_save_name, image_i)

    loss = loss_sum/(epoch_size*args.batch_size)
    print('Test epochs: {}\tLoss {:2.3f}'.format(epoch_size, loss))
    
    print('mean error and failure rate')
    landmark_error_norm = landmark_error/(epoch_size*args.batch_size)
    error_str ='mean error : {:2.3f}'.format(landmark_error_norm)
    failure_rate=landmark_01_num/(epoch_size*args.batch_size)
    failure_rate_str ='failure rate: L1 {:2.3f}'.format(failure_rate)
    print(error_str+'\n'+failure_rate_str+'\n')
    
def heatmap2landmark(heatmap):
    landmark = []
    h,w,c = heatmap.shape
    for i in range(c):
        m,n=divmod(np.argmax(heatmap[i]),w)
        landmark.append(n/w)
        landmark.append(m/h)
    return landmark
def save_image_example(sess, list_ops, args):
    save_nbatch = 10
    save_path = os.path.join(args.model_dir, 'image_example')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    image_batch, landmarks_batch, attribute_batch = list_ops['train_next_element']

    for b in range(save_nbatch):
        images, landmarks, attributes = sess.run([image_batch, landmarks_batch, attribute_batch])
        for i in range(images.shape[0]):
            img = images[i] * 256
            img = img.astype(np.uint8)
            if args.image_channels == 1:
                img = np.concatenate((img, img, img), axis=2)
            else:
                img = img[:, :, ::-1].copy()

            land = landmarks[i].reshape(-1, 2) * img.shape[:2]
            for x, y in land.astype(np.int32):
                cv2.circle(img, (x, y), 1, (0, 0, 255))
            save_name = os.path.join(save_path, '{}_{}.jpg'.format(b,i))
            cv2.imwrite(save_name, img)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_list', type=str,default='data/train_data/list.txt')
    parser.add_argument('--test_list', type=str, default='data/test_data/list.txt')
    parser.add_argument('--seed',type=int, default=666)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='models/')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_epoch', type=str, default='10,20,50,100,200,500')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--level', type=str, default='L5')
    parser.add_argument('--save_image_example',action='store_false')
    parser.add_argument('--debug', type=str, default='True')
    return parser.parse_args(argv)

if __name__ == '__main__':
    print(sys.argv)
    main(parse_arguments(sys.argv[1:]))

