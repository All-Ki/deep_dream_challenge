import random
import numpy as np
import cv2
from functools import partial
import tensorflow as tf
import sys
import urllib
import os
import zipfile

def T(graph, layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]

resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(sess, img, t_grad, t_input, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def render_deepdream(sess, t_obj, t_input, img0,
                     iter_n=10, step=1.5, octave_n=2, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

    # split the image into a number of octaves
    img = img0
    octaves = []
    for _ in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)

    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for _ in range(iter_n):
            g = calc_grad_tiled(sess, img, t_grad, t_input)
            img += g*(step / (np.abs(g).mean()+1e-7))

    #Step 5 return frame
    output_frame = img / 255.0
    output_frame = np.uint8(np.clip(output_frame, 0, 1)*255)
    return output_frame

def setup():
    #Step 1 - download google's pre-trained neural network
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = 'data/'
    model_name = os.path.split(url)[-1]

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    local_zip_file = os.path.join(data_dir, model_name)
    if not os.path.exists(local_zip_file):
        # Download
        model_url = urllib.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())
        # Extract
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    # start with a gray image with a little noise
    img_noise = np.random.uniform(size=(224,224,3)) + 100.0

    model_fn = 'tensorflow_inception_graph.pb'

    #Step 2 - Creating Tensorflow session and loading the model
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input':t_preprocessed})

    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))

   	#Step 3 - Pick a layer to enhance our image
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 129 # picking some feature channel to visualize

    return sess, graph, t_input

def run(sess, graph, t_input, input_filename):
    #open webcam
    cap = cv2.VideoCapture(0)

    writer = None
    i = 0
    LIM_FRAMES = 12
    while cap.isOpened() and i < LIM_FRAMES:
        ret, frame = cap.read()

        if frame is None:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = np.float32(frame)

        #Step 4 - Apply gradient ascent to that layer
        output_frame = render_deepdream(sess, tf.square(T(graph, 'mixed4c')), t_input, frame)
        if writer is None:
            frame_size = (output_frame.shape[1], output_frame.shape[0])
            writer = cv2.VideoWriter('output.avi', cv2.cv.FOURCC(*'XVID'), 4, frame_size)

        writer.write(output_frame)
        i += 1
        print 'frame %i complete.' % i

    cap.release()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        input_filename = 'input.avi'
    else:
        input_filename = sys.argv[1]

    sess, graph, t_input = setup()
    run(sess, graph, input_filename)

