from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import tensorflow as tf
import numpy as np
import cv2
import time

image_size = 112
out_dir = 'result'
image_file = './data/test_data/imgs/1_31_Waiter_Waitress_Waiter_Waitress_31_484_0.png'

from euler_angles_utils import calculate_pitch_yaw_roll
from euler_angles_utils import calculate_pitch_yaw_roll2

def test_model_ckpt(meta_file, ckpt_file):
    #meta_file = './models/model.meta'
    #ckpt_file = './models/model.ckpt-999'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            print('Loading feature extraction model.')
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(tf.get_default_session(), ckpt_file)

            graph = tf.get_default_graph()
            images_placeholder = graph.get_tensor_by_name('image_batch:0')
            phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')

            landmarks = graph.get_tensor_by_name('pfld_inference/fc/BiasAdd:0')

            image = cv2.imread(image_file)
            input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            input = cv2.resize(input, (image_size, image_size))
            input = input.astype(np.float32)/256.0
            input = np.expand_dims(input, 0)

            feed_dict = {
                images_placeholder: input,
                phase_train_placeholder: False
            }

            pre_landmarks = sess.run(landmarks, feed_dict=feed_dict)
            pre_landmark = pre_landmarks[0]

            h, w, _ = image.shape
            pre_landmark = pre_landmark.reshape(-1, 2) * [h, w]
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(image, (x, y), 1, (0, 255, 0))
            cv2.imshow('pfld', image)
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(out_dir, 'test.jpg'), image)

def test_model_freeze_graph(pb_fliename):
    #pb_path = 'deploy/pfld_freeze.pb'
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_fliename, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            images_placeholder = sess.graph.get_tensor_by_name("image_batch:0")
            #phase_train_placeholder = sess.graph.get_tensor_by_name('phase_train:0')
            landmarks = sess.graph.get_tensor_by_name('pfld_inference/fc/BiasAdd:0')

            image = cv2.imread(image_file)
            input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            input = cv2.resize(input, (image_size, image_size))
            input = input.astype(np.float32)/256.0
            input = np.expand_dims(input, 0)

            feed_dict = {
                images_placeholder: input,
                #phase_train_placeholder: False
            }

            start_time = time.time()
            pre_landmarks, pre_angles = sess.run(landmarks, feed_dict=feed_dict)
            get_landmarks_time = time.time() - start_time
            print('get_landmarks_time %f seconds' % get_landmarks_time)

            pre_landmark = pre_landmarks[0]

            h, w, _ = image.shape
            pre_landmark = pre_landmark.reshape(-1, 2) * [h, w]
            print(pre_landmark)
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(image, (x, y), 1, (0, 255, 0), thickness = -1)
            cv2.imshow('pfld', image)
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(out_dir, 'test.jpg'), image)

def test_model_lite(lite_filename):
    # load TFLite model and allocate tensors
    interpreter = tf.contrib.lite.Interpreter(model_path=lite_filename)
    interpreter.allocate_tensors()

    # get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # test model on input data
    image = cv2.imread(image_file)
    input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    input = cv2.resize(input, (image_size, image_size))
    input = input.astype(np.float32)/256.0
    input = np.expand_dims(input, 0)

    interpreter.set_tensor(input_details[0]['index'], input)
    interpreter.invoke()
    start_time = time.time()
    pre_landmarks = interpreter.get_tensor(output_details[0]['index'])
    get_landmarks_time = time.time() - start_time
    print('get_landmarks_time %f seconds' % get_landmarks_time)
    pre_landmark = pre_landmarks[0]

    h, w, _ = image.shape
    pre_landmark = pre_landmark.reshape(-1, 2) * [h, w]
    for (x, y) in pre_landmark.astype(np.int32):
        cv2.circle(image, (x, y), 1, (0, 255, 0))
    cv2.imshow('0', image)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(out_dir, 'test.jpg'), image)

if __name__ == '__main__':
    pb_path = 'deploy/pfld_freeze.pb'
    test_model_freeze_graph(pb_path)

