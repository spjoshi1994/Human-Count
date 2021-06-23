import time
import tensorflow as tf
import tensorflow.compat.v1 as tf
import os
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import argparse
#from tqdm import tqdm
import sys
import shutil
import subprocess
import glob
import json
import math
import operator
import matplotlib.pyplot as plt
from keras2tf import *

tf.enable_eager_execution()

num_confidence_scores = 14
num_class_probs = 7
PLOT_PROB_THRESH = 0.5  # 0.4 #55 #0.6
NMS_THRESH = 0.1
PROB_THRESH = 0.5
TOP_N_DETECTION = 10  # 64
BATCH_SIZE = 1
CLASSES = 1
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
MINOVERLAP = 0.5


def error(msg):
    print(msg)
    sys.exit(0)


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi



"""
 check if the number is a float between 0.0 and 1.0
"""


def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if 0.0 < val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False




"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""




def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


"""
 Draws text in image
"""



def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)


"""
 Plot - adjust axes
"""


def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


"""
 Draw plot using Matplotlib
"""


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()



def map(arguments):
    GT_PATH = os.path.abspath("inference_output/ground_truth/")
    DR_PATH = os.path.abspath("inference_output/predictions/")
    IMG_PATH = os.path.abspath("inference_output/image_output/")

    if os.path.exists(IMG_PATH):
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                # no image files found
                args.no_animation = False
    else:
        args.no_animation = False
    # try to import OpenCV if the user didn't choose the option --no-animation
    show_animation = False
    if not args.no_animation:
        show_animation = False
    # try to import Matplotlib if the user didn't choose the option --no-plot
    draw_plot = False
    if not args.no_plot:
        draw_plot = False

    """
     Create a ".temp_files/" and "output/" directory
    """
    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH):  # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)
    output_files_path = "output"
    if os.path.exists(output_files_path):  # if it exist already
        # reset the output directory
        shutil.rmtree(output_files_path)

    os.makedirs(output_files_path)
    if draw_plot:
        os.makedirs(os.path.join(output_files_path, "classes"))
    if show_animation:
        os.makedirs(os.path.join(output_files_path, "images", "detections_one_by_one"))

    """
     ground-truth
         Load each of the ground-truth files into a temporary ".json" file.
         Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}
    gt_files = []
    for txt_file in ground_truth_files_list:
        # print(txt_file)
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, _, _, _, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <0.0> <0.0> <0.0> <left> <top> <right> <bottom>\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)

            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        # dump bounding_boxes into a ".json" file
        new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    """
     detection-results
         Load each of the detection-results files into a temporary ".json" file.
    """
    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            # print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    # print("match")
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
                    # print(bounding_boxes)
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
     Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    # open file to store the output
    with open(output_files_path + "/output.txt", 'w') as output_file:
        output_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            """
             Load detection-results of that class
            """
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            """
             Assign detection-results to ground-truth objects
            """
            nd = len(dr_data)
            tp = [0] * nd  # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                if show_animation:
                    # find ground truth image
                    ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                    # tifCounter = len(glob.glob1(myPath,"*.tif"))
                    if len(ground_truth_img) == 0:
                        error("Error. Image not found with id: " + file_id)
                    elif len(ground_truth_img) > 1:
                        error("Error. Multiple image with id: " + file_id)
                    else:  # found image
                        # print(IMG_PATH + "/" + ground_truth_img[0])
                        # Load image
                        img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                        # load image with draws of multiple detections
                        img_cumulative_path = output_files_path + "/images/" + ground_truth_img[0]
                        if os.path.isfile(img_cumulative_path):
                            img_cumulative = cv2.imread(img_cumulative_path)
                        else:
                            img_cumulative = img.copy()
                        # Add bottom border to image
                        bottom_border = 60
                        BLACK = [0, 0, 0]
                        img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                # assign detection-results to ground truth object if any
                # open ground-truth with that file_id
                gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                              + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign detection as true positive/don't care/false positive
                if show_animation:
                    status = "NO MATCH FOUND!"  # status is only used in the animation
                # set minimum overlap
                min_overlap = MINOVERLAP
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                            if show_animation:
                                status = "MATCH!"
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                            if show_animation:
                                status = "REPEATED MATCH!"
                else:
                    # false positive
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

                """
                 Draw image to show animation
                """
                if show_animation:
                    height, widht = img.shape[:2]
                    # colors (OpenCV works with BGR)
                    white = (255, 255, 255)
                    light_blue = (255, 200, 100)
                    green = (0, 255, 0)
                    light_red = (30, 30, 255)
                    # 1st line
                    margin = 10
                    v_pos = int(height - margin - (bottom_border / 2.0))
                    text = "Image: " + ground_truth_img[0] + " "
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue,
                                                         line_width)
                    if ovmax != -1:
                        color = light_red
                        if status == "INSUFFICIENT OVERLAP":
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                        else:
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                            color = green
                        img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                    # 2nd line
                    v_pos += int(bottom_border / 2.0)
                    rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                    text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                        float(detection["confidence"]) * 100)
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    color = light_red
                    if status == "MATCH!":
                        color = green
                    text = "Result: " + status + " "
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if ovmax > 0:  # if there is intersections between the bounding-boxes
                        bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                        cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font, 0.6, light_blue, 1,
                                    cv2.LINE_AA)
                    bb = [int(i) for i in bb]
                    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    cv2.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                    # show image
                    cv2.imshow("Animation", img)
                    cv2.waitKey(20)  # show for 20 ms
                    # save image to output
                    output_img_path = output_files_path + "/images/detections_one_by_one/" + class_name + "_detection" + str(
                        idx) + ".jpg"
                    cv2.imwrite(output_img_path, img)
                    # save the image with all the objects drawn to it
                    cv2.imwrite(img_cumulative_path, img_cumulative)

            # print(tp)
            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            # print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            # print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            # print(prec)

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_AP += ap
            text = "{0:.2f}%".format(
                ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
            """
             Write to output.txt
            """
            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
            if not args.quiet:
                print(text)
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr

            """
             Draw plot
            """
            if draw_plot:
                plt.plot(rec, prec, '-o')
                # add a new penultimate point to the list (mrec[-2], 0.0)
                # since the last line segment (and respective area) do not affect the AP value
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                # set window title
                fig = plt.gcf()  # gcf - get current figure
                fig.canvas.set_window_title('AP ' + class_name)
                # set plot title
                plt.title('class: ' + text)
                # plt.suptitle('This is a somewhat long figure title', fontsize=16)
                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                # Alternative option -> wait for button to be pressed
                # while not plt.waitforbuttonpress(): pass # wait for key display
                # Alternative option -> normal display
                # plt.show()
                # save the plot
                fig.savefig(output_files_path + "/classes/" + class_name + ".png")
                plt.cla()  # clear axes for next plot

        if show_animation:
            cv2.destroyAllWindows()

        output_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes

        text = "mAP = {0:.2f}%".format(mAP * 100)
        output_file.write(text + "\n")
        print(text)

    """
     Draw false negatives
    """
    pink = (203, 192, 255)
    for tmp_file in gt_files:
        ground_truth_data = json.load(open(tmp_file))
        # print(ground_truth_data)
        # get name of corresponding image
        start = TEMP_FILES_PATH + '/'
        img_id = tmp_file[tmp_file.find(start) + len(start):tmp_file.rfind('_ground_truth.json')]
        img_cumulative_path = output_files_path + "/images/" + img_id + ".jpg"
        img = cv2.imread(img_cumulative_path)
        if img is None:
            img_path = IMG_PATH + '/' + img_id + ".jpg"
            img = cv2.imread(img_path)
        # draw false negatives
        for obj in ground_truth_data:
            if not obj['used']:
                bbgt = [int(round(float(x))) for x in obj["bbox"].split()]
                cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), pink, 2)
        cv2.imwrite(img_cumulative_path, img)

    # remove the temp_files directory
    shutil.rmtree(TEMP_FILES_PATH)

    """
     Count total of detection-results
    """
    # iterate through all the files
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        # get lines to list
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            # count that object
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                det_counter_per_class[class_name] = 1
    # print(det_counter_per_class)
    dr_classes = list(det_counter_per_class.keys())

    """
     Plot the total number of occurences of each class in the ground-truth
    """
    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = output_files_path + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
        )

    """
     Write number of ground-truth objects per class to results.txt
    """
    with open(output_files_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            output_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    """
     Finish counting true positives
    """
    for class_name in dr_classes:
        # if class exists in detection-result but not in ground-truth then there are no true positives in that class
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0
    # print(count_true_positives)

    """
     Plot the total number of occurences of each class in the "detection-results" folder
    """
    if draw_plot:
        window_title = "detection-results-info"
        # Plot title
        plot_title = "detection-results\n"
        plot_title += "(" + str(len(dr_files_list)) + " files and "
        count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
        plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
        # end Plot title
        x_label = "Number of objects per class"
        output_path = output_files_path + "/detection-results-info.png"
        to_show = False
        plot_color = 'forestgreen'
        true_p_bar = count_true_positives
        draw_plot_func(
            det_counter_per_class,
            len(det_counter_per_class),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            true_p_bar
        )

    """
     Write number of detected objects per class to output.txt
    """
    with open(output_files_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            output_file.write(text)

    """
     Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = output_files_path + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )

    """
     Draw mAP plot (Show AP's of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP * 100)
        x_label = "Average Precision"
        output_path = output_files_path + "/mAP.png"
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )






def set_anchors():
    H, W, B = 14, 14, 7  # 224/16=14, 7 anchors
    div_scale = 2.0 * 1  # 224/224=1

    anchor_shapes = np.reshape(
        [np.array(
            [
                [int(368. / div_scale), int(368. / div_scale)],
                [int(276. / div_scale), int(276. / div_scale)],
                [int(184. / div_scale), int(184. / div_scale)],
                [int(138. / div_scale), int(138. / div_scale)],
                [int(92. / div_scale), int(92. / div_scale)],
                [int(69. / div_scale), int(69. / div_scale)],
                [int(46. / div_scale), int(46. / div_scale)]])] * H * W,
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
    def __init__(self, model_path,arguments):
        self.input_shape = [3, 224, 224]
        self.graph = tf.Graph()
        input_tensor_name = "input:0"
        #output_tensor_name = "conv12/convolution:0"
        output_tensor_name = "output_node0:0"
        #input_tensor_name = arguments.input_node
        #output_tensor_name = arguments.output_node
        self.channels = arguments.channels

        self.CLASS_NAMES = ('Person')
        self.cls2clr = {'Person': (255, 191, 0)}

        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                self.load_model('inference.pb')
                self.input = tf.get_default_graph().get_tensor_by_name(input_tensor_name)
                self.output = tf.get_default_graph().get_tensor_by_name(output_tensor_name)
                # print(self.input, self.output)

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
                # print("J {} OV {} Threshold {}".format(j, ov, threshold))
                if ov > threshold:
                    keep[order[j + i + 1]] = False
        return keep

    def filter_prediction(self, boxes, probs, cls_idx):
        if len(probs.numpy()) > TOP_N_DETECTION > 0:
            order = probs.numpy().argsort()[:-TOP_N_DETECTION - 1:-1]
            # print(type(probs), type(order))
            probs = probs.numpy()[order]
            # print(probs)
            boxes = boxes.numpy()[order]
            cls_idx = cls_idx.numpy()[order]
        else:
            print("===="*10)
            filtered_idx = np.nonzero(probs > PROB_THRESH)[0]
            probs = probs.numpy()[filtered_idx]
            boxes = boxes.numpy()[filtered_idx]
            cls_idx = cls_idx.numpy()[filtered_idx]

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
        # print(preds[:,:,:,7:14])


        if self.channels == '1':
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

        else:
        	 pred_class_probs = tf.reshape(
	              tf.nn.softmax(
	                  tf.reshape(
	                      preds[:, :, :, 7:14],
	                      [-1, CLASSES]
	                  )
	              ),
	              [BATCH_SIZE, ANCHORS, CLASSES]
	          )

	         pred_conf = tf.sigmoid(
	              tf.reshape(
	                  preds[:, :, :, :7],
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
        probs = tf.multiply(
            pred_class_probs,
            tf.reshape(pred_conf, [BATCH_SIZE, ANCHORS, 1]))

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
            xmin, ymin, xmax, ymax = int(bbox[0]*scale_width), int(bbox[1]*scale_height), int(bbox[2]*scale_width), int(bbox[3]*scale_height)
            l = label.split(':')[0]  # text before "CLASS: (PROB)"
            if cdict and l in cdict:
                c = cdict[l]
            else:
                c = color
            # draw box
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 2 * scale)
            cv2.rectangle(im, (xmin, ymin - 30),(xmin+(len(label)*12), ymin-5), c, -1)
            cv2.putText(im, label, (xmin, ymin - 10),0, 0.75, (255,255,255),2)

def main(arguments):
    model = sholder_surfing_model(arguments.pb,arguments)
    mode = 'image'
    if mode == 'image':
        out_root_path = os.path.abspath("./inference_output")
        out_pred_root_path = os.path.join(out_root_path, "predictions")
        out_image_root_path = os.path.join(out_root_path, "image_output")
        ground_truth_path = os.path.join(out_root_path,"ground_truth")
        dir_list = [out_root_path, out_pred_root_path, out_image_root_path,ground_truth_path]
        for directory in dir_list:
            if not os.path.exists(directory):
                os.mkdir(directory)
        input_images = arguments.dataset_path


        test_set_name = 'ImageSets/'+ arguments.test_set + '.txt' 



        test_images_name = os.path.join(input_images , test_set_name)

        file1 = open(test_images_name, 'r')
        Lines = file1.readlines()
        #image_path = input_images + '/training/images'
        #label_path = input_images + "/training/labels"
        image_path = os.path.join(input_images, "training/images")
        label_path = os.path.join(input_images ,"training/labels")





        #with tqdm(total=len(input_images), file=sys.stdout) as pbar:
        for line in Lines:
            image_name = line.strip()+".jpg"
            ground_truth_name = line.strip()+".txt"
            shutil.copy(os.path.join(label_path,ground_truth_name),ground_truth_path)

            image = cv2.imread(os.path.join(image_path, image_name))
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image_copy = image.copy()

            if arguments.channels == '1':
            	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            	image = image.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 1)) / 128
            else:
            	image = image.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 3)) / 128

            #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            #image = image.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 3)) / 128
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
                           [model.CLASS_NAMES[idx] + ': (%.2f)' % prob for idx, prob in zip(final_class, final_probs)],
                           cdict=model.cls2clr)
            out_image_path = os.path.join(out_image_root_path, image_name)
            output_label_path = os.path.join(out_pred_root_path, image_name.split(".")[0]+'.txt')
            with open(output_label_path, "a+") as file:
                for label in out_label_list:
                    kitti_format = ""
                    for element in label:
                        if kitti_format == "":
                            kitti_format += str('Person')
                        else:
                            kitti_format += " " + str(element)
                    file.write(kitti_format + "\n")
            cv2.imwrite(out_image_path, image_copy)
        map(arguments)
        shutil.rmtree(out_root_path, ignore_errors = False)
        shutil.rmtree('./output/', ignore_errors = False)
            #pbar.update(1)
    else:
        cv2.startWindowThread()
        cv2.namedWindow("Frame")
        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        vs = cv2.VideoCapture(0)
        width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x_scale = width/IMAGE_WIDTH
        y_scale = height/IMAGE_HEIGHT
        while True:
            start_time = time.time()
            ret, frame = vs.read()
            image_copy = frame.copy()
            image = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image = image.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, 3)) / 128
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
    parser.add_argument("-i", "--dataset_path", required=False, default="./images", help="Input dataset Path")
    #parser.add_argument("-o", "--output", required=False, default="./inference_output", help="Output Path")
    parser.add_argument("-inp", "--input_node", required=False, default="batch:0", help="input_node_name")
    parser.add_argument("-out", "--output_node", required=False, default="conv12/bias_add:0", help="output_node_name")
    parser.add_argument("-m", "--mode", required=False, default='image', help="camera or image")
    parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
    parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
    parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    parser.add_argument('-c', '--channels', help="number of channels '1' or '3'.",default='3')
    parser.add_argument('-t', '--test_set', help="Calculate mAP on 'test' or 'val'.", required=False, default = "val")
    # parser.add_argument('-gpu','--gpu_option',action="store_true")
    # parser.add_argument('-mt','--metrics_option',action="store_true")



    args = parser.parse_args()
    convert('inference.pb', args.pb)
    main(args)

    #subprocess.call("python3 main.py -i inference_output/image_output/ -g inference_output/ground_truth/ -p inference_output/predictions/ -na -np", shell=True)
