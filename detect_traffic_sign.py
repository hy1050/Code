import sys
import tensorflow as tf
import numpy as np
import model
import argparse
import os
import util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import config


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_fn', help='image filename')
    parser.add_argument(
        '--save_img',
        action='store_true',
        default=False,
        help='Use this flag if you want to save result image (default: False)')
    return parser.parse_args()


def traffic_sign_recognition(sess, img, obj_proposal, graph_params):
    # recognition results
    recog_results = {}
    recog_results['obj_proposal'] = obj_proposal

    # Resize image
    if img.shape != model.IMG_SHAPE:
        img = cv2.resize(img, (model.IMG_WIDTH, model.IMG_HEIGHT))

    # Pre-processing(Hist equalization)
    print(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    print(type(img))
    split_img = np.array(cv2.split(img))
    print(type(split_img))
    split_img[0] = cv2.equalizeHist(split_img[0])
    eq_img = cv2.merge(split_img)
    eq_img = cv2.cvtColor(eq_img, cv2.COLOR_YCrCb2BGR)
    # Scaling in [0, 1]
    eq_img = (eq_img / 255.).astype(np.float32)
    eq_img = np.expand_dims(eq_img, axis=0)

    # Traffic sign recognition
    pred = sess.run(
        [graph_params['pred']],
        feed_dict={graph_params['target_image']: eq_img})
    recog_results['pred_class'] = np.argmax(pred)
    recog_results['pred_prob'] = np.max(pred)

    return recog_results


def setup_graph():
    graph_params = {}
    graph_params['graph'] = tf.Graph()
    with graph_params['graph'].as_default():
        model_params = model.params()
        
        graph_params['target_image'] = tf.compat.v1.placeholder(
            tf.float32,
            shape=(1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_CHANNELS))
        logits = model.cnn(
            graph_params['target_image'], model_params, keep_prob=0.5)
        graph_params['pred'] = tf.nn.softmax(logits)
        graph_params['saver'] = tf.compat.v1.train.Saver()
    return graph_params


def cls2name(cls):
    SIGNNAMES_FILE = 'signnames.csv'
    signnames_ = np.loadtxt(
        os.path.join(config.VNTSR_ROOT_DIR, SIGNNAMES_FILE),
        delimiter=',',
        dtype=np.str)
    # skip first row
    signnames = signnames_[1:]
    # dictionary that convert class number to sign name
    to_name = {s[0]: s[1] for s in signnames}

    # convert class name to signname
    name = to_name[str(cls)]
    return name


def main():
    # args = parse_cmdline()
    # print(args.img_fn)
    # img_fn = os.path.abspath(args.img_fn)
    img_fn = r'C:\Users\BHG81HC\Documents\Private_Hy\Graduate_Thesis\Documents\Model_CNN\test1.jpg'
    # print(img_fn)
    # save_img = args.save_img
    if not os.path.exists(img_fn):
        print('Not found: {}'.format(img_fn))
        sys.exit(-1)
    else:
        print('Target image: {}'.format(img_fn))

    # Load target image
    target_image = cv2.imread(img_fn)

    # Get object proposals
    object_proposals = util.get_object_proposals(target_image)

    # Setup computation graph
    graph_params = setup_graph()

    # Model initialize
    sess = tf.compat.v1.Session(graph=graph_params['graph'])
    print(sess)
    tf.compat.v1.global_variables_initializer()
    if os.path.exists('models'):
        save_path = os.path.join('models', 'deep_traffic_sign_model')
        graph_params['saver'].restore(sess, save_path)
        print('Model restored')
    else:
        print('Initialized')

    # traffic sign recognition
    results = []
    for obj_proposal in object_proposals:
        x, y, w, h = obj_proposal
        crop_image = target_image[y:y + h, x:x + w]
        results.append(
            traffic_sign_recognition(sess, crop_image, obj_proposal,
                                     graph_params))
    """
    del_idx = []
    for i, result in enumerate(results):
        if result['pred_class'] == common.CLASS_NAME[-1]:
            del_idx.append(i)
    results = np.delete(results, del_idx)
    """
    # Non-max suppression
    nms_results = util.nms(results, pred_prob_th=0.999999, iou_th=0.4)

    # Draw rectangles on the target image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))

    for result in nms_results:
        print(result)
        (x, y, w, h) = result['obj_proposal']
        ax.text(
            x,
            y,
            cls2name(result['pred_class']),
            fontsize=13,
            bbox=dict(facecolor='red', alpha=0.7))
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()

    # save the target image
    save_fname = os.path.splitext(os.path.basename(img_fn))[0] + '_result.jpg'
    if save_img:
        fig.savefig(save_fname, bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    main()
