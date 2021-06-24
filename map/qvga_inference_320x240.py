import argparse
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
from tqdm import tqdm

tf.enable_eager_execution()

num_confidence_scores = 14
num_class_probs = 7
PLOT_PROB_THRESH = 0.5
NMS_THRESH = 0.1
PROB_THRESH = 0.5
TOP_N_DETECTION = 10
BATCH_SIZE = 1
CLASSES = 1
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240


def set_anchors():
    H, W, B = 15, 20, 7
    div_scale_h = 2.0 * (224 / IMAGE_HEIGHT)
    div_scale_w = 2.0 * (224 / IMAGE_WIDTH)

    anchor_shapes = np.reshape(
        [np.array(
            [
                [int(368. / div_scale_h), int(368. / div_scale_w)],
                [int(276. / div_scale_h), int(276. / div_scale_w)],
                [int(184. / div_scale_h), int(184. / div_scale_w)],
                [int(138. / div_scale_h), int(138. / div_scale_w)],
                [int(92. / div_scale_h), int(92. / div_scale_w)],
                [int(69. / div_scale_h), int(69. / div_scale_w)],
                [int(46. / div_scale_h), int(46. / div_scale_w)]])] * H * W,
        (H, W, B, 2)
    )

    center_x = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, W + 1) * float(IMAGE_WIDTH) / (W + 1)] * H * B),
                (B, H, W)
            ),
            (1, 2, 0)
        ),
        (H, W, B, 1)
    )
    center_y = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, H + 1) * float(IMAGE_HEIGHT) / (H + 1)] * W * B),
                (B, W, H)
            ),
            (2, 1, 0)
        ),
        (H, W, B, 1)
    )
    anchors = np.reshape(
        np.concatenate((center_x, center_y, anchor_shapes), axis=3),
        (-1, 4)
    )

    return anchors


ANCHOR_BOX = set_anchors()
ANCHORS = len(ANCHOR_BOX)


class sholder_surfing_model(object):
    def __init__(self, model_path):
        self.graph = tf.Graph()
        output_tensor_name = "output_node0:0"
        input_tensor_name = "input:0"

        self.CLASS_NAMES = ('person')
        self.cls2clr = {'person': (255, 191, 0)}

        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                print(tf.get_default_graph())
                self.load_model(model_path)
                self.input = tf.get_default_graph().get_tensor_by_name(input_tensor_name)
                self.output = tf.get_default_graph().get_tensor_by_name(output_tensor_name)
                print("Model loaded")

    def get_inference_output(self, data):
        feed_dict = {self.input: data}
        return self.sess.run(self.output, feed_dict=feed_dict)

    @staticmethod
    def load_model(model):
        with gfile.FastGFile(model, 'rb') as file_:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_.read())
            tf.import_graph_def(graph_def, name='')

    @staticmethod
    def bbox_transform(bbox):
        cx, cy, w, h = bbox
        out_box = [[]] * 4
        out_box[0] = cx - w / 2
        out_box[1] = cy - h / 2
        out_box[2] = cx + w / 2
        out_box[3] = cy + h / 2
        return out_box

    @staticmethod
    def bbox_transform_inv(bbox):
        xmin, ymin, xmax, ymax = bbox
        out_box = [[]] * 4
        width = xmax - xmin + 1.0
        height = ymax - ymin + 1.0
        out_box[0] = xmin + 0.5 * width
        out_box[1] = ymin + 0.5 * height
        out_box[2] = width
        out_box[3] = height
        return out_box

    @staticmethod
    def batch_iou(boxes, box):
        lr = np.maximum(
            np.minimum(boxes[:, 0] + 0.5 * boxes[:, 2], box[0] + 0.5 * box[2]) - \
            np.maximum(boxes[:, 0] - 0.5 * boxes[:, 2], box[0] - 0.5 * box[2]),
            0
        )
        tb = np.maximum(
            np.minimum(boxes[:, 1] + 0.5 * boxes[:, 3], box[1] + 0.5 * box[3]) - \
            np.maximum(boxes[:, 1] - 0.5 * boxes[:, 3], box[1] - 0.5 * box[3]),
            0
        )
        inter = lr * tb
        union = boxes[:, 2] * boxes[:, 3] + box[2] * box[3] - inter
        return inter / union

    def nms(self, boxes, probs, threshold):
        order = probs.argsort()[::-1]
        keep = [True] * len(order)

        for i in range(len(order) - 1):
            ovps = self.batch_iou(boxes[order[i + 1:]], boxes[order[i]])
            for j, ov in enumerate(ovps):
                if ov > threshold:
                    keep[order[j + i + 1]] = False
        return keep

    def filter_prediction(self, boxes, probs, cls_idx):
        if len(probs.eval(session=tf.Session())) > TOP_N_DETECTION > 0:
            order = probs.eval(session=tf.Session()).argsort()[:-TOP_N_DETECTION - 1:-1]
            probs = probs.eval(session=tf.Session())[order]
            boxes = boxes.eval(session=tf.Session())[order]
            cls_idx = cls_idx.eval(session=tf.Session())[order]
        else:
            filtered_idx = np.nonzero(probs > PROB_THRESH)[0]
            probs = probs.eval(session=tf.Session())[filtered_idx]
            boxes = boxes.eval(session=tf.Session())[filtered_idx]
            cls_idx = cls_idx.eval(session=tf.Session())[filtered_idx]

        final_boxes = []
        final_probs = []
        final_cls_idx = []

        idx_per_class = [i for i in range(len(probs))]
        keep = self.nms(boxes[idx_per_class], probs[idx_per_class], NMS_THRESH)
        for i in range(len(keep)):
            if keep[i]:
                final_boxes.append(boxes[idx_per_class[i]])
                final_probs.append(probs[idx_per_class[i]])
                final_cls_idx.append(cls_idx[i])
        return final_boxes, final_probs, final_cls_idx

    def interpreat_output(self, preds):
        pred_class_probs = tf.reshape(
            tf.nn.softmax(
                tf.reshape(
                    preds[:, :, :, :7],
                    [-1, CLASSES]
                )
            ),
            [BATCH_SIZE, ANCHORS, CLASSES]
        )

        pred_conf = tf.sigmoid(
            tf.reshape(
                preds[:, :, :, 7:14],
                [BATCH_SIZE, ANCHORS]
            )
        )

        pred_box_delta = tf.reshape(preds[:, :, :, 14:], [BATCH_SIZE, ANCHORS, 4])
        delta_x, delta_y, delta_w, delta_h = tf.unstack(pred_box_delta, axis=2)

        anchor_x = ANCHOR_BOX[:, 0]
        anchor_y = ANCHOR_BOX[:, 1]
        anchor_w = ANCHOR_BOX[:, 2]
        anchor_h = ANCHOR_BOX[:, 3]

        box_center_x = tf.identity(anchor_x + delta_x * anchor_w)
        box_center_y = tf.identity(anchor_y + delta_y * anchor_h)
        box_width = tf.identity(anchor_w * delta_w)
        box_height = tf.identity(anchor_h * delta_h)
        xmins, ymins, xmaxs, ymaxs = self.bbox_transform(
            [box_center_x, box_center_y, box_width, box_height])

        xmins = tf.minimum(tf.maximum(0.0, xmins), IMAGE_WIDTH - 1.0)
        ymins = tf.minimum(tf.maximum(0.0, ymins), IMAGE_HEIGHT - 1.0)
        xmaxs = tf.maximum(tf.minimum(IMAGE_WIDTH - 1.0, xmaxs), 0.0)
        ymaxs = tf.maximum(tf.minimum(IMAGE_HEIGHT - 1.0, ymaxs), 0.0)

        det_boxes = tf.transpose(
            tf.stack(self.bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
            (1, 2, 0))
        # probs = tf.multiply(
        #     pred_class_probs,
        #     tf.reshape(pred_conf, [BATCH_SIZE, ANCHORS, 1]))
        probs = tf.reshape(pred_conf, [BATCH_SIZE, ANCHORS, 1])

        det_probs = tf.reduce_max(probs, 2)
        det_class = tf.argmax(probs, 2)
        final_boxes, final_probs, final_class = self.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx = [idx for idx in range(len(final_probs)) if final_probs[idx] > PLOT_PROB_THRESH]
        final_probs = [final_probs[idx] for idx in keep_idx]
        keep_list = keep_idx
        final_probs = [final_probs[idx] for idx in keep_list]
        final_boxes = [final_boxes[idx] for idx in keep_list]
        final_class = [final_class[idx] for idx in keep_list]
        return final_boxes, final_class, final_probs

    def draw_box(self, im, box_list, scale_width, scale_height, label_list, color=(128, 0, 128), cdict=None, scale=1):
        for bbox, label in zip(box_list, label_list):
            xmin, ymin, xmax, ymax = int(bbox[0] * scale_width), int(bbox[1] * scale_height), int(
                bbox[2] * scale_width), int(bbox[3] * scale_height)
            l = label.split(':')[0]  # text before "CLASS: (PROB)"
            if cdict and l in cdict:
                c = cdict[l]
            else:
                c = color
            # draw box
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 2 * scale)
            cv2.rectangle(im, (xmin, ymin - 30), (xmin + (len(label) * 12), ymin - 5), c, -1)
            cv2.putText(im, label, (xmin, ymin - 10), 0, 0.75, (255, 255, 255), 2)


def main(arguments):
    model = sholder_surfing_model(arguments.pb)
    if arguments.mode == 'image':
        out_root_path = os.path.abspath(arguments.output)
        out_pred_root_path = os.path.join(out_root_path, "predictions")
        out_image_root_path = os.path.join(out_root_path, "image_output")
        dir_list = [out_root_path, out_pred_root_path, out_image_root_path]
        for directory in dir_list:
            if not os.path.exists(directory):
                os.mkdir(directory)
        input_images = os.listdir(arguments.input_images)
        with tqdm(total=len(input_images), file=sys.stdout) as pbar:
            for image_name in input_images:
                image = cv2.imread(os.path.join(arguments.input_images, image_name))
                image_copy = image.copy()
                height, width, c = image_copy.shape
                x_scale = width / IMAGE_WIDTH
                y_scale = height / IMAGE_HEIGHT
                image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 1)) / 128
                out_tensor = model.get_inference_output(np.array([image]))
                final_boxes, final_class, final_probs = model.interpreat_output(out_tensor)
                out_bbox_list = []
                out_label_list = []
                for label_index in range(len(final_boxes)):
                    box = model.bbox_transform(final_boxes[label_index])
                    out_bbox_list.append(box)
                    label = [model.CLASS_NAMES[final_class[label_index]], final_probs[label_index]]
                    for cord in box:
                        label.append(cord)
                    out_label_list.append(label)
                model.draw_box(image_copy, out_bbox_list, 1, 1,
                               [model.CLASS_NAMES[idx] + ': (%.2f)' % prob for idx, prob in
                                zip(final_class, final_probs)],
                               cdict=model.cls2clr)
                out_image_path = os.path.join(out_image_root_path, image_name)
                output_label_path = os.path.join(out_pred_root_path, image_name.split(".")[0] + '.txt')
                # print(label)
                with open(output_label_path, "a+") as file:
                    for label in out_label_list:
                        kitti_format = ""
                        for element in label:
                            if kitti_format == "":
                                kitti_format += str('person')
                            else:
                                kitti_format += " " + str(element)
                        file.write(kitti_format + "\n")
                cv2.imwrite(out_image_path, image_copy)
                pbar.update(1)
    else:
        cv2.startWindowThread()
        cv2.namedWindow("Frame")
        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        vs = cv2.VideoCapture(0)
        width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x_scale = width / IMAGE_WIDTH
        y_scale = height / IMAGE_HEIGHT
        while True:
            start_time = time.time()
            ret, frame = vs.read()
            image_copy = frame.copy()
            image = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 1)) / 128
            out_tensor = model.get_inference_output(np.array([image]))
            final_boxes, final_class, final_probs = model.interpreat_output(out_tensor)
            out_bbox_list = []
            out_label_list = []
            for label_index in range(len(final_boxes)):
                box = model.bbox_transform(final_boxes[label_index])
                out_bbox_list.append(box)
                label = [model.CLASS_NAMES[final_class[label_index]], final_probs[label_index]]
                for cord in box:
                    label.append(cord)
                out_label_list.append(label)
            model.draw_box(image_copy, out_bbox_list, x_scale, y_scale,
                           [model.CLASS_NAMES[idx] + ': (%.2f)' % prob for idx, prob in zip(final_class, final_probs)],
                           cdict=model.cls2clr)
            cv2.imshow("Frame", image_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pb", required=True, help="Input Model Path")
    parser.add_argument("-i", "--input_images", required=False, default="./images", help="Input Images Path")
    parser.add_argument("-o", "--output", required=False, default="./inference_output", help="Output Path")
    parser.add_argument("-m", "--mode", required=False, default='camera', help="camera or image")

    args = parser.parse_args()
    main(args)
