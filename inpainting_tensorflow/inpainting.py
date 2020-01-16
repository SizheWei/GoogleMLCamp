import numpy as np
import cv2
import os
import subprocess
import glob
import tensorflow as tf
from util.util import generate_mask_rect, generate_mask_stroke
from net.network import GMCNNModel
import argparse
import os
import time
import dlib
import sys,os,traceback
import numpy as np
from face import Makeup


class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataset', type=str, default='paris_streetview',required=False,
                                 help='The dataset of the experiment.')
        self.parser.add_argument('--data_file', type=str, default='./imgs/celebahq_512x512',required=False, help='the file storing testing file paths')
        self.parser.add_argument('--test_dir', type=str, default='./test_results',required=False, help='models are saved here')
        self.parser.add_argument('--load_model_dir', type=str, default="./checkpoints/celebahq_512x512_freeform/",required=False, help='pretrained models are given here')
        self.parser.add_argument('--seed', type=int, default=1,required=False, help='random seed')
        self.parser.add_argument('--gpu_ids', type=str,required=False, default='0')

        self.parser.add_argument('--model', type=str,required=False, default='gmcnn')
        self.parser.add_argument('--random_mask', type=int, default=0,required=False,
                                 help='using random mask')

        self.parser.add_argument('--img_shapes', type=str, default='512,512,3',required=False,
                                 help='given shape parameters: h,w,c or h,w')
        self.parser.add_argument('--mask_shapes', type=str, default='32,32',required=False,
                                 help='given mask parameters: h,w')
        self.parser.add_argument('--mask_type', type=str, default='rect',required=False)
        self.parser.add_argument('--test_num', type=int, default=-1,required=False)
        self.parser.add_argument('--mode', type=str, default='save',required=False)
        self.parser.add_argument('--phase', type=str, default='test',required=False)

        # for generator
        self.parser.add_argument('--g_cnum', type=int, default=32,required=False,
                                 help='# of generator filters in first conv layer')
        self.parser.add_argument('--d_cnum', type=int, default=32,required=False,
                                 help='# of discriminator filters in first conv layer')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(args=[])

        if self.opt.data_file != '':
            self.opt.dataset_path = self.opt.data_file

        if os.path.exists(self.opt.test_dir) is False:
            os.mkdir(self.opt.test_dir)

        assert self.opt.random_mask in [0, 1]
        self.opt.random_mask = True if self.opt.random_mask == 1 else False

        assert self.opt.mask_type in ['rect', 'stroke']

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        str_mask_shapes = self.opt.mask_shapes.split(',')
        self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        # model name and date
        self.opt.date_str = 'test_'+time.strftime('%Y%m%d-%H%M%S')
        self.opt.model_folder = self.opt.date_str + '_' + self.opt.dataset + '_' + self.opt.model
        self.opt.model_folder += '_s' + str(self.opt.img_shapes[0]) + 'x' + str(self.opt.img_shapes[1])
        self.opt.model_folder += '_gc' + str(self.opt.g_cnum)
        self.opt.model_folder += '_randmask-' + self.opt.mask_type if self.opt.random_mask else ''
        if self.opt.random_mask:
            self.opt.model_folder += '_seed-' + str(self.opt.seed)
        self.opt.saving_path = os.path.join(self.opt.test_dir, self.opt.model_folder)

        if os.path.exists(self.opt.saving_path) is False and self.opt.mode == 'save':
            os.mkdir(self.opt.saving_path)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt


config = TestOptions().parse()

config.dataset_path="./imgs/data/"
config.mask_type='stroke'
pathfile = glob.glob(os.path.join(config.dataset_path, '*.jpg'))
test_num = len(pathfile)
for i in range(test_num):
    image = cv2.imread(pathfile[i])
    h, w = image.shape[:2]

    if h >= config.img_shapes[0] and w >= config.img_shapes[1]:
        h_start = (h-config.img_shapes[0]) // 2
        w_start = (w-config.img_shapes[1]) // 2
        image = image[h_start: h_start+config.img_shapes[0], w_start: w_start+config.img_shapes[1], :]
    else:
        t = min(h, w)
        image = image[(h-t)//2:(h-t)//2+t, (w-t)//2:(w-t)//2+t, :]
        image = cv2.resize(image, (config.img_shapes[1], config.img_shapes[0]))
    cv2.imwrite(os.path.join(config.dataset_path, '{:03d}.png'.format(i)),image)
pathfile = glob.glob(os.path.join(config.dataset_path, '*.png'))
test_num= len(pathfile)
print(test_num)

model = GMCNNModel()

reuse = False
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = False
sess=tf.Session(config=sess_config)

input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])

output = model.evaluate(input_image_tf, input_mask_tf, config=config, reuse=reuse)
output = (output + 1) * 127.5
output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
output = tf.cast(output, tf.uint8)

# load pretrained model
vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                      vars_list))
sess.run(assign_ops)
print('Model loaded.')
total_time = 0

if config.random_mask:
    np.random.seed(config.seed)


mu = Makeup()