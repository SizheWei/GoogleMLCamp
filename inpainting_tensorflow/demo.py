from tkinter import *
from PIL import Image, ImageTk, ImageDraw
import tkinter.filedialog as tkFileDialog
import numpy as np
import cv2
import os
import subprocess
import argparse
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
from changemask import changemouth
from changemask import changenose
from changemask import changehead
from changemask import changeeye

class Paint():
    MARKER_COLOR = 'white'

    def __init__(self,model):

        self.root = Tk()
        self.inpaintingmodel=model

        self.rect_button = Button(self.root, text='rectangle', command=self.use_rect, width=12, height=3)
        self.rect_button.grid(row=0, column=2)

        self.poly_button = Button(self.root, text='stroke', command=self.use_poly, width=12, height=3)
        self.poly_button.grid(row=1, column=2)

        self.revoke_button = Button(self.root, text='revoke', command=self.revoke, width=12, height=3)
        self.revoke_button.grid(row=2, column=2)

        self.clear_button = Button(self.root, text='clear', command=self.clear, width=12, height=3)
        self.clear_button.grid(row=3, column=2)

        self.c = Canvas(self.root, bg='white', width=512+8, height=512)
        self.c.grid(row=0, column=0, rowspan=8)

        self.out = Canvas(self.root, bg='white', width=512+8, height=512)
        self.out.grid(row=0, column=1, rowspan=8)

        self.save_button = Button(self.root, text="save", command=self.save, width=12, height=3)
        self.save_button.grid(row=6, column=2)

        self.load_button = Button(self.root, text='load', command=self.load, width=12, height=3)
        self.load_button.grid(row=5, column=2)

        self.fill_button = Button(self.root, text='fill', command=self.fill, width=12, height=3)
        self.fill_button.grid(row=7, column=2)
        self.filename = None

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.eraser_on = False
        self.active_button = self.rect_button
        self.isPainting = False
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.c.bind('<Button-1>', self.beginPaint)
        self.c.bind('<Enter>', self.icon2pen)
        self.c.bind('<Leave>', self.icon2mice)
        self.mode = 'rect'
        self.rect_buf = None
        self.line_buf = None
        assert self.mode in ['rect', 'poly']
        self.paint_color = self.MARKER_COLOR
        self.mask_candidate = []
        self.rect_candidate = []
        self.im_h = None
        self.im_w = None
        self.mask = None
        self.result = None
        self.blank = None
        self.line_width = 15

        self.reuse = False


    def checkResp(self):
        assert len(self.mask_candidate) == len(self.rect_candidate)

    def load(self):
        self.filename = tkFileDialog.askopenfilename(initialdir='./imgs',
                                                     title="Select file",
                                                     filetypes=(("png files", "*.png"), ("jpg files", "*.jpg"),
                                                                ("all files", "*.*")))
        self.filename_ = self.filename.split('/')[-1][:-4]
        self.filepath = '/'.join(self.filename.split('/')[:-1])
        #print(self.filename_, self.filepath)
        try:
            photo = Image.open(self.filename)
            self.image = cv2.imread(self.filename)
        except:
            print('do not load image')
        else:
            self.im_w, self.im_h = photo.size
            self.mask = np.zeros((self.im_h, self.im_w, 1)).astype(np.uint8)
            print(photo.size)
            self.displayPhoto = photo
            self.displayPhoto = self.displayPhoto.resize((self.im_w, self.im_h))
            self.draw = ImageDraw.Draw(self.displayPhoto)
            self.photo_tk = ImageTk.PhotoImage(image=self.displayPhoto)
            self.c.create_image(0, 0, image=self.photo_tk, anchor=NW)
            self.rect_candidate.clear()
            self.mask_candidate.clear()
            if self.blank is None:
                self.blank = Image.open('./imgs/blank.png')
            self.blank = self.blank.resize((self.im_w, self.im_h))
            self.blank_tk = ImageTk.PhotoImage(image=self.blank)
            self.out.create_image(0, 0, image=self.blank_tk, anchor=NW)

    def save(self):
        img = np.array(self.displayPhoto)
        #cv2.imwrite(os.path.join(self.filepath, 'tmp.png'), img)

        if self.mode == 'rect':
            self.mask[:,:,:] = 0
            for rect in self.mask_candidate:
                self.mask[rect[1]:rect[3], rect[0]:rect[2], :] = 1
        image = np.expand_dims(img, 0)
        mask = np.expand_dims(self.mask, 0)
        result_tmp = self.inpaintingmodel.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})
        result = cv2.merge([result_tmp[0][:, :, 0], result_tmp[0][:, :, 1], result_tmp[0][:, :, 2]])
        #cv2.imwrite(os.path.join(self.filepath, self.filename_+'_mask.png'), self.mask*255)
        cv2.imwrite(os.path.join(self.filepath, self.filename_+'_out.png'), result)
        photo = Image.open(os.path.join(self.filepath, self.filename_+'_out.png'))
        self.displayPhotoResult = photo
        self.displayPhotoResult = self.displayPhotoResult.resize((self.im_w, self.im_h))
        self.photo_tk_result = ImageTk.PhotoImage(image=self.displayPhotoResult)
        self.out.create_image(0, 0, image=self.photo_tk_result, anchor=NW)
        # cv2.imwrite(os.path.join(self.filepath, self.filename_+'_out.png'), result[0][:, :, ::-1])
        #cv2.imwrite(os.path.join(self.filepath, self.filename_ + '_gm_result.png'), self.result[0][:, :, ::-1])
        #self.displayPhotoResult = photo

    def fill(self):
        image = cv2.imread(os.path.join(self.filepath, self.filename))
        # cv2.imwrite(os.path.join(self.filepath, 'tmp.png'), img)
        im, temp_bgr, faces = mu.read_and_mark(os.path.join(self.filepath, self.filename))
        mask=self.mask.copy()
        mask[:,:,0]=0
        maskeye = mask.copy()
        path = os.path.join(self.filepath, self.filename)
        face = next(iter(faces[path]))
        facemask = face.organs['left eye'].get_mask_abs() + face.organs['right eye'].get_mask_abs()
        for p in range(512):
            for q in range(512):
                if facemask[p][q].min() > 0:
                    # pass
                    maskeye[p][q] = 1
        maskeye = changeeye(maskeye)
        headmask = face.organs['left brow'].get_mask_abs() + face.organs['right brow'].get_mask_abs()
        maskhead = mask.copy()
        masknose = mask.copy()
        maskmouth = mask.copy()
        for p in range(512):
            for q in range(512):
                if headmask[p][q].min() > 0:
                    # pass
                    maskhead[p][q] = 1
        maskhead = changehead(maskhead)

        for p in range(512):
            for q in range(512):
                if maskhead[p][q] == 1 or maskeye[p][q] == 1:
                    mask[p][q] = 1

        nosemask = face.organs['nose'].get_mask_abs()
        for p in range(512):
            for q in range(512):
                if nosemask[p][q].min() > 0:
                    # pass
                    masknose[p][q] = 1

        masknose = changenose(masknose)

        for p in range(512):
            for q in range(512):
                if masknose[p][q] == 1:
                    mask[p][q] = 0

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        result_tmp = self.inpaintingmodel.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})
        result = cv2.merge([result_tmp[0][:, :, 2], result_tmp[0][:, :, 1], result_tmp[0][:, :, 0]])
        cv2.imwrite(os.path.join(self.filepath, self.filename_ + '_autoout.png'), result)
        photo = Image.open(os.path.join(self.filepath, self.filename_ + '_autoout.png'))
        self.displayPhotoResult = photo
        self.displayPhotoResult = self.displayPhotoResult.resize((self.im_w, self.im_h))
        self.photo_tk_result = ImageTk.PhotoImage(image=self.displayPhotoResult)
        self.out.create_image(0, 0, image=self.photo_tk_result, anchor=NW)
        # if self.mode == 'rect':
        #     self.mask[:, :, :] = 0
        #     for rect in self.mask_candidate:
        #         self.mask[rect[1]:rect[3], rect[0]:rect[2], :] = 1
        # image = np.expand_dims(img, 0)
        # mask = np.expand_dims(self.mask, 0)
        # result_tmp = self.inpaintingmodel.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})
        # result = cv2.merge([result_tmp[0][:, :, 0], result_tmp[0][:, :, 1], result_tmp[0][:, :, 2]])
        # # cv2.imwrite(os.path.join(self.filepath, self.filename_+'_mask.png'), self.mask*255)
        # cv2.imwrite(os.path.join(self.filepath, self.filename_ + '_out.png'), result)
        # photo = Image.open(os.path.join(self.filepath, self.filename_ + '_out.png'))
        # self.displayPhotoResult = photo
        # self.displayPhotoResult = self.displayPhotoResult.resize((self.im_w, self.im_h))
        # self.photo_tk_result = ImageTk.PhotoImage(image=self.displayPhotoResult)
        # self.out.create_image(0, 0, image=self.photo_tk_result, anchor=NW)

    def use_rect(self):
        self.activate_button(self.rect_button)
        self.mode = 'rect'

    def use_poly(self):
        self.activate_button(self.poly_button)
        self.mode = 'poly'

    def revoke(self):
        if len(self.rect_candidate) > 0:
            self.c.delete(self.rect_candidate[-1])
            self.rect_candidate.remove(self.rect_candidate[-1])
            self.mask_candidate.remove(self.mask_candidate[-1])
        self.checkResp()

    def clear(self):
        self.mask = np.zeros((self.im_h, self.im_w, 1)).astype(np.uint8)
        if self.mode == 'poly':
            photo = Image.open(self.filename)
            self.image = cv2.imread(self.filename)
            self.displayPhoto = photo
            self.displayPhoto = self.displayPhoto.resize((self.im_w, self.im_h))
            self.draw = ImageDraw.Draw(self.displayPhoto)
            self.photo_tk = ImageTk.PhotoImage(image=self.displayPhoto)
            self.c.create_image(0, 0, image=self.photo_tk, anchor=NW)
        else:
            if self.rect_candidate is None or len(self.rect_candidate) == 0:
                return
            for item in self.rect_candidate:
                self.c.delete(item)
            self.rect_candidate.clear()
            self.mask_candidate.clear()
            self.checkResp()

    #TODO: reset canvas
    #TODO: undo and redo
    #TODO: draw triangle, rectangle, oval, text

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def beginPaint(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.isPainting = True

    def paint(self, event):
        if self.start_x and self.start_y and self.mode == 'rect':
            self.end_x = max(min(event.x, self.im_w), 0)
            self.end_y = max(min(event.y, self.im_h), 0)
            rect = self.c.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, fill=self.paint_color)
            if self.rect_buf is not None:
                self.c.delete(self.rect_buf)
            self.rect_buf = rect
        elif self.old_x and self.old_y and self.mode == 'poly':
            line = self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                                      width=self.line_width, fill=self.paint_color, capstyle=ROUND,
                                      smooth=True, splinesteps=36)
            cv2.line(self.mask, (self.old_x, self.old_y), (event.x, event.y), (1), self.line_width)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None
        if self.mode == 'rect':
            self.isPainting = False
            rect = self.c.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y,
                                           fill=self.paint_color)
            if self.rect_buf is not None:
                self.c.delete(self.rect_buf)
            self.rect_buf = None
            self.rect_candidate.append(rect)

            x1, y1, x2, y2 = min(self.start_x, self.end_x), min(self.start_y, self.end_y),\
                             max(self.start_x, self.end_x), max(self.start_y, self.end_y)
            # up left corner, low right corner
            self.mask_candidate.append((x1, y1, x2, y2))
            print(self.mask_candidate[-1])

    def icon2pen(self, event):
        return

    def icon2mice(self, event):
        return

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
Paint(sess)