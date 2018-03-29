import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import numpy as np

from glob import glob

import imageio

imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

import scipy.misc

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    vgg_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3 = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4 = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7 = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input, vgg_keep_prob, vgg_layer3, vgg_layer4, vgg_layer7


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, sess=None, vgg_input=None, keep_prob=None):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # based on the lesson 10: Scene Understanding, FCN-8 - Decoder

    stddev = 0.01
    reg = 1e-3

    # 1x1 convolution to the layer 7 of VGG-16. Output 5 18
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes,
                                  kernel_size=1,
                                  strides=1,
                                  padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))

    # 1x1 convolution to the layer 4 of VGG-16. Output 10 36
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes,
                                  kernel_size=1,
                                  strides=1,
                                  padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))

    # 1x1 convolution to the layer 3 of VGG-16. Output 20 72
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes,
                                  kernel_size=1,
                                  strides=1,
                                  padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))

    # Apply decoder and upsample output to 10 36
    layer7_decoder = tf.layers.conv2d_transpose(layer7_1x1, num_classes,
                                                kernel_size=4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))

    # Add skip connection to layer 4 of VGG. Output 10 36
    skip_connection_l7_decoder_l4_1x1 = tf.add(layer7_decoder, layer4_1x1)

    # Apply decoder and upsample output to 20 72
    layer4_decoder = tf.layers.conv2d_transpose(skip_connection_l7_decoder_l4_1x1, num_classes,
                                                kernel_size=4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))

    # Add skip connection to layer 3 of VGG. Output 20 72
    skip_connection_l4_decoder_l3_1x1 = tf.add(layer4_decoder, layer3_1x1)

    # Apply decoder and upsample output to 160 576
    out_mid_layer = tf.layers.conv2d_transpose(skip_connection_l4_decoder_l3_1x1, num_classes,
                                               kernel_size=4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))

    nn_last_layer = tf.layers.conv2d_transpose(out_mid_layer, num_classes,
                                               kernel_size=8,
                                               strides=4,
                                               padding='same',
                                               kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))

    if sess is not None:
        img = np.random.rand(8, 160, 576, 3)
        prints = [
            tf.Print(nn_last_layer, [tf.shape(nn_last_layer), " ------------------- OUT LAYER -------------------"],
                     summarize=4)]
        sess.run(tf.global_variables_initializer())
        sess.run(prints, feed_dict={vgg_input: img, keep_prob: 1.0})

    return nn_last_layer


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = adam_optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        print("Epoch: %d" % epoch)

        total_loss = 0

        idx = 0

        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label,
                                          keep_prob: 0.75, learning_rate: 0.00001})
            idx += 1
            total_loss += loss

        print("Average loss: %f" % (total_loss / idx))


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    model_path = "models/model.ckpt"

    load_model = False

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches

        data_folder = os.path.join(data_dir, 'data_road/training')

        get_batches_fn = helper.gen_batch_function(data_folder, image_shape)

        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))

        print("Dataset: %d images" % len(image_paths))

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        vgg_input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)

        print("----------------------------------VGG------------------------------------------")
        print(tf.trainable_variables())

        nn_last_layer = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes,
                               sess=sess, vgg_input=vgg_input, keep_prob=keep_prob)

        print("----------------------------------LAYERS------------------------------------------")
        print(tf.trainable_variables())

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function

        epochs = 50

        # dataset contains 289 images, so batch must be as small as possible
        batch_size = 1

        if load_model:
            saver = tf.train.Saver()

            saver.restore(sess, model_path)
        else:
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_input,
                     correct_label, keep_prob, learning_rate)

            saver = tf.train.Saver()

            saver.save(sess, model_path)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, vgg_input)

        # OPTIONAL: Apply the trained model to a video

        if not load_model:
            saver.save(sess, model_path)

        def process_image(image):
            image = scipy.misc.imresize(image, image_shape)

            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, vgg_input: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            return np.array(street_im)

        print("Processing the video")

        output_path = 'videos/project_video_processed.mp4'
        clip2 = VideoFileClip('videos/project_video.mp4')
        yellow_clip = clip2.fl_image(process_image)
        yellow_clip.write_videofile(output_path, audio=False)

        print("Video was processed.")


if __name__ == '__main__':
    run()
