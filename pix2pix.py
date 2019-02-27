from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import cv2
import scipy.io

parser = argparse.ArgumentParser()
#parser.add_argument("--input_dir", help="path to folder containing images")
#parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
#parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=30, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=50000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
#parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
# commented by (kjh)
# parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

# a.mode="train"
# a.output_dir="ped_train"
# # a.input_dir="D:/imageSet/image_for_pix2pix/train10/images_only_ped"
# a.which_direction="AtoB"
# input_paths = glob.glob(os.path.join("D:/imageSet/image_for_pix2pix/train2/images_with_bboxes/image_segment", "*.png"))
# bbox_paths = "D:/imageSet/image_for_pix2pix/train2/images_with_bboxes/bbox"
# N = len(input_paths)
# feat_stride=8

a.mode="test"
a.output_dir="ped_test"
a.input_dir="D:/imageSet/image_for_pix2pix/test20(stantard)/images_with_ped"
a.checkpoint="ped_train"
input_paths = glob.glob(os.path.join("D:/imageSet/image_for_pix2pix/test20(standard)/images_pad", "*.png"))
bbox_paths = "D:/imageSet/image_for_pix2pix/train2/images_with_bboxes/bbox"
N = len(input_paths)
feat_stride=8

EPS = 1e-12
# commented by (kjh)
# CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "inputs, targets")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, seg_result, cls_result, bbox_result,"
                                        "discrim_loss_GAN, discrim_loss_segment, discrim_loss_cls, discrim_loss_bbox, discrim_grads_and_vars,"
                                        "gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")
Bbox = collections.namedtuple("Bbox", "labels, label_weights, bbox_targets, bbox_loss_weights")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        # image = tf.squeeze(image,0)
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def discrim_conv(batch_input, out_channels, stride, k_size):
    # 좌우상하에 1씩 padding (batch_size, channel 에는 padding 하지 않음)
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=k_size, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    # batch normalization over channels? (kjh)
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def load_examples(raw_input):

    raw_input.set_shape([None, None, 3])
    # raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
    if a.lab_colorization:
        # load color and brightness from image, no B image exists here
        lab = rgb_to_lab(raw_input)
        L_chan, a_chan, b_chan = preprocess_lab(lab)
        a_images = tf.expand_dims(L_chan, axis=2)
        b_images = tf.stack([a_chan, b_chan], axis=2)
    else:
        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]

        a_images = preprocess(raw_input[:, :width // 3, :])
        b_images = tf.expand_dims(preprocess(raw_input[:, width // 3:width * 2 // 3, 0]), axis=2)
        c_images = tf.expand_dims(raw_input[:, width * 2 // 3:, 0], axis=2)
    if a.which_direction == "AtoB":
        inputs, targets, segment = [a_images, b_images, c_images]
    elif a.which_direction == "BtoA":
        inputs, targets, segment = [b_images, a_images, c_images]
    else:
        raise Exception("invalid direction")

    with tf.name_scope("input_images"):
        input_images = tf.image.resize_images(inputs, [512, 640], method=tf.image.ResizeMethod.AREA)
        input_images = tf.image.per_image_standardization(input_images)
    with tf.name_scope("target_images"):
        target_images = tf.image.resize_images(targets, [512, 640], method=tf.image.ResizeMethod.AREA)
    with tf.name_scope("segment_images"):
        segment_images = tf.image.resize_images(segment, [512//feat_stride, 640//feat_stride], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return tf.expand_dims(input_images, 0), tf.expand_dims(target_images, 0), tf.expand_dims(segment_images, 0)


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 512, 512, in_channels] => [batch, 256, 256, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        a.ngf * 8,
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 8, 0.0),
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:

                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            # 첫번째 decoder 에서 6x4 --> 5x4 로 나타내기 위해 1x2 pooling 사용 (kjh)
            # if decoder_layer == 0:
            #     output = tf.nn.avg_pool(output, [1, 1, 2, 1], [1, 1, 1, 1], 'VALID')

            output = batchnorm(output)
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets, segments, labels, label_weights, bbox_targets, bbox_loss_weights):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2, k_size=4)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                # stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                if i == n_layers - 1:
                    convolved = discrim_conv(layers[-1], out_channels, stride=1, k_size=3)
                else:
                    convolved = discrim_conv(layers[-1], out_channels, stride=2, k_size=4)
                # convolved = discrim_conv(layers[-1], out_channels, stride=2, k_size=4)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1, k_size=3)
            output = tf.sigmoid(convolved)
            layers.append(output)

        with tf.variable_scope("layer_seg"):
            convolved = discrim_conv(rectified, out_channels=1, stride=1, k_size=3)
            seg_output = tf.sigmoid(convolved)

        with tf.variable_scope("layer_RPN"):
            convolved = discrim_conv(rectified, out_channels=512, stride=1, k_size=3)
            rectified = lrelu(convolved, 0.2)
            with tf.variable_scope("cls"):
                cls_output = tf.layers.conv2d(rectified, filters=9, kernel_size=1, strides=(1, 1),
                                 padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
                # cls_output = discrim_conv(rectified, out_channels=9, stride=1, k_size=1)
                cls_output = tf.sigmoid(cls_output)
            with tf.variable_scope("bbox"):
                bbox_output = tf.layers.conv2d(rectified, filters=36, kernel_size=1, strides=(1, 1),
                                 padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
                # bbox_output = discrim_conv(rectified, out_channels=36, stride=1, k_size=1)

        # [-1] indicates last index (kjh)
        return layers[-1], seg_output, cls_output, bbox_output

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(tf.image.resize_images(inputs, [512, 512], method=tf.image.ResizeMethod.AREA), out_channels)
        # image resize with BILINEAR 만 gradient 가 전파됨
        outputs = tf.image.resize_images(outputs, [512, 640], method=tf.image.ResizeMethod.BILINEAR)
    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real, _, _, _ = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake, seg, cls, bbox = create_discriminator(inputs, outputs)
            # predict_ped = (kjh)
    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        # sigmoid 처리 하기전의 값을 logits 에 넣음
        discrim_loss_GAN = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        discrim_loss_segment = tf.reduce_mean(-(segments * tf.log(seg + EPS) + (1 - segments) * tf.log(1 - seg + EPS)))
        # loss = -w[zlog(s)+(1-z)log(1-s)] (kjh)
        discrim_loss_cls = tf.reduce_mean(-label_weights * (labels*tf.log(cls+EPS) + (1-labels)*tf.log(1-cls+EPS)))
        # simply use abs loss rather than smooth L1 loss in fast rcnn (kjh)
        discrim_loss_bbox = tf.losses.absolute_difference(bbox_targets, bbox, bbox_loss_weights)
        discrim_loss = discrim_loss_GAN + discrim_loss_cls + discrim_loss_bbox + discrim_loss_segment

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        # segmentation loss를 generator에 줌으로써, pedestrian을 더 잘분별할수 있는 이미지를 생성하도록 함
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        # gen_loss = gen_loss_GAN + gen_loss_L1 * 100
        # gen_loss = gen_loss_GAN + gen_loss_L1 + discrim_loss_cls + discrim_loss_bbox + discrim_loss_segment
        gen_loss = gen_loss_GAN + discrim_loss_cls + discrim_loss_bbox + discrim_loss_segment
    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    # update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
    update_losses = ema.apply([discrim_loss_GAN, discrim_loss_segment, discrim_loss_cls, discrim_loss_bbox, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        seg_result=seg,
        cls_result=cls,
        bbox_result=bbox,
        outputs=outputs,

        discrim_loss_GAN=ema.average(discrim_loss_GAN),
        discrim_loss_segment=ema.average(discrim_loss_segment),
        discrim_loss_cls=ema.average(discrim_loss_cls),
        discrim_loss_bbox=ema.average(discrim_loss_bbox),
        discrim_grads_and_vars=discrim_grads_and_vars,

        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,

        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, name, model_name=None, step=None):
    image_dir = os.path.join(a.output_dir, "images"+'-'+model_name)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    fileset = {"name": name, "step": step}
    for kind in ["inputs", "outputs", "targets", "segment"]:
        filename = name + "-" + kind + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][0]
        with open(out_path, "wb") as f:
            f.write(contents)
    filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def sample_minibatch(bbox):
    fg_rois_per_image = 20
    bg_rois_per_image = 100
    bg_weight = 1

    ex_asign_labels = bbox[:, 0]
    labels = np.zeros(np.shape(bbox)[0])
    label_weights = np.zeros(np.shape(bbox)[0])
    bbox_targets = bbox[:, 1:]
    bbox_loss_weights = bbox_targets * 0

    # labels=bbox[:,1]
    fg_idx = np.where(np.equal(ex_asign_labels, 1))
    bg_idx = np.where(np.equal(ex_asign_labels, -1))

    fg_num = np.minimum(np.size(fg_idx), fg_rois_per_image)
    bg_num = np.minimum(np.size(bg_idx), bg_rois_per_image)
    np.random.shuffle(fg_idx)
    np.random.shuffle(bg_idx)
    fg_idx = fg_idx[:fg_num]
    bg_idx = bg_idx[:bg_num]

    if fg_num != 0:
        labels[fg_idx] = ex_asign_labels[fg_idx]
        label_weights[fg_idx] = fg_rois_per_image / fg_num
        bbox_loss_weights[fg_idx, :] = fg_rois_per_image / fg_num

    label_weights[bg_idx] = bg_weight

    labels = np.expand_dims(np.transpose(np.reshape(labels, [640//feat_stride, 512//feat_stride, 9]), (1, 0, 2)), 0)
    label_weights = np.expand_dims(np.transpose(np.reshape(label_weights, [640//feat_stride, 512//feat_stride, 9]), (1, 0, 2)), 0)
    bbox_targets = np.expand_dims(np.transpose(np.reshape(bbox_targets, [640//feat_stride, 512//feat_stride, 36]), (1, 0, 2)), 0)
    bbox_loss_weights = np.expand_dims(np.transpose(np.reshape(bbox_loss_weights, [640//feat_stride, 512//feat_stride, 36]), (1, 0, 2)), 0)

    return labels, label_weights, bbox_targets, bbox_loss_weights


def add_noise(image, name, mode):
    image = image.astype(np.float32) / 255
    if mode=="train":
        set = int(name[3:5])
        # only for daytime image
        if set < 3:
            noise = np.random.normal(0, 0.1, [512, 640, 3])
            image[:512, :640, :] = image[:512, :640, :] + noise
            image[image > 1] = 1
            image[image < 0] = 0

    return image


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)

        # commented by (kjh)
        # disable these features in test mode
        # a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)

        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
        # convert grayscale to RGB
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)

        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        # commented by (kjh)
        # input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = deprocess(create_generator(preprocess(batch_input), 3))

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    img = tf.placeholder(tf.float32)
    # daytime = tf.placeholder(tf.bool)
    input_img, target_img, segment_img = load_examples(img)

    lbs = tf.placeholder(tf.float32)
    lbsw = tf.placeholder(tf.float32)
    bboxt = tf.placeholder(tf.float32)
    bboxlw = tf.placeholder(tf.float32)

    steps_per_epoch = int(N / a.batch_size)
    print("examples count = %d" % N)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(input_img, target_img, segment_img, lbs, lbsw, bboxt, bboxlw)

    inputs = deprocess(input_img)
    targets = deprocess(target_img)
    outputs = deprocess(model.outputs)

    def convert(image):
        # commented by (kjh)
        # if a.aspect_ratio != 1.0:
        #     # upscale to correct aspect ratio
        #     size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
        image = tf.image.resize_images(image, size=[512,640], method=tf.image.ResizeMethod.BICUBIC)
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("convert_segment"):
        converted_segment = convert(model.seg_result)

    with tf.name_scope("encode_images"):
        display_fetches = {
            # "paths": examples.paths,
            # tf.map_fn: tf.image.encode_png 함수를 모든 converted_inputs 에 대해 적용
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
            "segment": tf.map_fn(tf.image.encode_png, converted_segment, dtype=tf.string, name="segment_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("segment_summary"):
        tf.summary.image("segment", converted_segment)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss_GAN", model.discrim_loss_GAN)
    tf.summary.scalar("discriminator_loss_segment", model.discrim_loss_segment)
    tf.summary.scalar("discriminator_loss_cls", model.discrim_loss_cls)
    tf.summary.scalar("discriminator_loss_bbox", model.discrim_loss_bbox)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    # max_to_keep: how many latest models will you save, to save all of them, set it to 'None' (kjh)
    saver = tf.train.Saver(max_to_keep=None)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        # if a.checkpoint is not None:
            # print("loading model from checkpoint")
            # checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            # saver.restore(sess, checkpoint)
            # saver.restore(sess, 'D:/pix2pix/ped_train/model-300000')

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            for m in range(400000//a.save_freq):
                if m==400000//a.save_freq-1:
                    checkpoint = tf.train.latest_checkpoint(a.checkpoint)
                    saver.restore(sess, checkpoint)
                    model_name = 'last'
                else:
                    model_name = str(a.save_freq*(m+1))
                    saver.restore(sess, 'D:/pix2pix/ped_train/model-'+model_name)

                start = time.time()
                max_steps = min(steps_per_epoch, max_steps)
                for step in range(max_steps):
                    image = cv2.imread(input_paths[step])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    name, _ = os.path.splitext(os.path.basename(input_paths[step]))
                    image = add_noise(image,name,a.mode)
                    # test 시에 thermal image 필요없으며 실제로 사용되지 않지만, 편의를 위해 함께 넣음
                    # [display_fetches, model.cls_result, model.bbox_result]
                    display_result, cls_output, bbox_output = sess.run([display_fetches, model.cls_result, model.bbox_result], feed_dict={img: image})
                    filesets = save_images(display_result, name, model_name=model_name)
                    for i, f in enumerate(filesets):
                        print("evaluated image", f["name"])
                    index_path = append_index(filesets)

                    bbox_dir = os.path.join(a.output_dir, "detection_result"+'-'+model_name)
                    if not os.path.exists(bbox_dir):
                        os.makedirs(bbox_dir)
                    bbox_result={"cls": cls_output, "bbox": bbox_output}
                    bbox_path = os.path.join(bbox_dir,name)
                    scipy.io.savemat(bbox_path, bbox_result)

                print("wrote index at", index_path)
                print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()

            L = list(range(N))
            random.seed(1)

            random.shuffle(L)
            n = 0

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                if n == N:
                    random.shuffle(L)
                    n = 0

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss_GAN"] = model.discrim_loss_GAN
                    fetches["discrim_loss_segment"] = model.discrim_loss_segment
                    fetches["discrim_loss_cls"] = model.discrim_loss_cls
                    fetches["discrim_loss_bbox"] = model.discrim_loss_bbox
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                # load image and gt
                image = cv2.imread(input_paths[L[n]])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                name, _ = os.path.splitext(os.path.basename(input_paths[L[n]]))
                image = add_noise(image, name, a.mode)
                mat = scipy.io.loadmat(os.path.join(bbox_paths, name))
                labels, label_weights, bbox_targets, bbox_loss_weights = sample_minibatch(mat['bbox_targets'])
                # fetches에 있는 모든 Tensor들을 실행하기위해 꼭 필요한 그래프 단편을 실행
                results = sess.run(fetches, options=options, run_metadata=run_metadata,
                                   feed_dict={img: image, lbs: labels, lbsw:label_weights, bboxt:bbox_targets, bboxlw:bbox_loss_weights})
                # results = sess.run(fetches, options=options, run_metadata=run_metadata)

                n = n + 1

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], name, step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / steps_per_epoch)
                    train_step = (results["global_step"] - 1) % steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss_GAN", results["discrim_loss_GAN"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("discrim_loss_segment", results["discrim_loss_segment"])
                    print("discrim_loss_cls", results["discrim_loss_cls"])
                    print("discrim_loss_bbox", results["discrim_loss_bbox"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)


                if sv.should_stop():
                    break


main()
