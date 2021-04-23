import os
from PIL import Image
import cv2
import numpy as np
import joblib
import time
import argparse

#####################################
# utils
#####################################
def read_img(img_path):
    '''return img with cv2 & 3ch'''
    try:
        img = cv2.imread(img_path)
        if img.shape == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    except:
        img = np.asarray(Image.open(img_path))
        if img.shape == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def get_xy(coord):
    '''handle both list and tuple'''
    return coord[0], coord[1]


def get_now_next_3_coord_and_idxs(contour, idx):
    '''return now_coord, [next1_coord, next2_coord, next3_coord]'''
    now_coord = contour[idx][0]
    next3_coord_list = list()
    next3_coord_idxs = list()
    while len(next3_coord_list) < 3:
        if idx + 1 >= len(contour):
            idx = 0
        else:
            idx = idx + 1
        next3_coord_list.append(contour[idx][0])
        next3_coord_idxs.append(idx)
    return now_coord, next3_coord_list, next3_coord_idxs


def compute_line_len(coord1, coord2, stair_type):
    coord1_x, coord1_y = get_xy(coord1)
    coord2_x, coord2_y = get_xy(coord2)
    if stair_type == "TOP_BOTTOM":
        return abs(coord2_x - coord1_x)
    else: # stair_type == "LEFT_RIGHT"
        return abs(coord2_y - coord1_y)


def weighted_average(val1, val2, weight1, weight2):
    # return (weight1*val1 + weight2*val2) // (weight1 + weight2)
    return int(round((weight1*val1 + weight2*val2) / float(weight1 + weight2)))


def draw_dot(mask_img, contours, DOT_SIZE, COLOR):
    '''draw "coordinates" as dot-shape'''
    for contour in contours:
        for coord in contour:
            mask_img = cv2.circle(mask_img, tuple(coord[0]), DOT_SIZE, COLOR, -1)
    return mask_img


#####################################
# sub_logic
#####################################
def is_diagonal(now_coord, prev_coord):
    '''find diagonal case (The rows and columns of both coordinates do not match)'''
    now_x, now_y = now_coord
    prev_x, prev_y = prev_coord
    if now_x == prev_x or now_y == prev_y:
        return False
    else:
        return True

def get_rectilinear_coord(now_coord, prev_coord):
    '''diagonal => rectilinear'''
    now_x, now_y = now_coord
    prev_x, prev_y = prev_coord

    # compute the direction of step
    if now_x > prev_x:
        direction_x = -1
    else:
        direction_x = 1
    if now_y > prev_y:
        direction_y = -1
    else:
        direction_y = 1

    # compute 1) step, 2) first step axis(x or y)
    is_change_x = None
    new_coord_list = list()
    diff_x = abs(now_x - prev_x)
    diff_y = abs(now_y - prev_y)
    if diff_x <= diff_y:
        step_y = direction_y * max(1, (diff_y // (diff_x + 1)))
        step_x = direction_x
        is_change_x = False
    else:  # diff_x > diff_y
        step_x = direction_x * max(1, (diff_x // (diff_y + 1)))
        step_y = direction_y
        is_change_x = True

    # move and save coord
    new_x, new_y = now_x, now_y
    while new_x != prev_x and new_y != prev_y:
        if is_change_x:
            new_x += step_x
        else:
            new_y += step_y
        is_change_x = not is_change_x
        new_coord_list.append((new_x, new_y))

    return new_coord_list[::-1]

  
def remove_unnecessary_points(contour, init_idx=0):
    '''
        Delete an mid pixel on the same line
        ex) (0,0) (0,10), (0,20) => delete (0,10)
        ex) (0,10) (123,10), (6574,10) => delete (123,10)
    '''
    del_count = 0
    if len(contour) >= 3:  # min_nums_of_points
        idx = init_idx
        while True:
            # get prev_coord, now_coord, next_coord
            now_coord, prev_coord = contour[idx][0], contour[idx - 1][0]
            if idx + 1 >= len(contour):
                next_coord = contour[0][0]
            else:
                next_coord = contour[idx+1][0]

            # find unnecessary case: same row or column
            if (prev_coord[0] == now_coord[0] == next_coord[0]) or (prev_coord[1] == now_coord[1] == next_coord[1]):
                del contour[idx]
                del_count += 1
                # end while-1
                if len(contour) < 3:
                    break
                else:
                    if idx >= 1:
                        idx = idx - 1
            else:
                idx = idx + 1

            # end while-2
            if idx >= len(contour):
                break
            else:
                continue
    return del_count

  
def remove_unnecessary_points_range(contour, init_idx=0):
    '''
        Delete an mid pixel on the same line
        ex) (0,0) (0,10), (0,20) => delete (0,10)
        ex) (0,10) (123,10), (6574,10) => delete (123,10)
    '''
    del_count = 0
    check_count = 0
    if len(contour) >= 3:  # min_nums_of_points
        idx = init_idx

        while True:
            check_count += 1

            # get prev_coord, now_coord, next_coord
            now_coord, prev_coord = contour[idx][0], contour[idx - 1][0]
            if idx + 1 >= len(contour):
                next_coord = contour[0][0]
            else:
                next_coord = contour[idx + 1][0]

            # find unnecessary case: same row or column
            if (prev_coord[0] == now_coord[0] == next_coord[0]) or (prev_coord[1] == now_coord[1] == next_coord[1]):
                del contour[idx]
                del_count += 1
                # end while-1
                if len(contour) < 3:
                    break
            else:
                idx = idx + 1

            if idx >= len(contour):
                idx = 0

            # end while-2
            if check_count >= 4:
                break
            else:
                continue
    return del_count

  
def find_stair_shape(now_coord, next3_coord_list, MAX_DISTANCE):
    '''Find coordinates consisting of 's' shapes'''
    next1_coord, next2_coord, next3_coord = next3_coord_list[0], next3_coord_list[1], next3_coord_list[2]
    now_x, now_y = get_xy(now_coord)
    next1_x, next1_y = get_xy(next1_coord)
    next2_x, next2_y = get_xy(next2_coord)
    next3_x, next3_y = get_xy(next3_coord)

    # case: top, bottom
    if now_y == next1_y and next1_x == next2_x and next2_y == next3_y \
            and ((now_x > next1_x and next2_x > next3_x) or (now_x < next1_x and next2_x < next3_x)) \
            and abs(next2_y-next1_y) <= MAX_DISTANCE:
        return "TOP_BOTTOM"
    # case: left, right
    elif now_x == next1_x and next1_y == next2_y and next2_x == next3_x \
            and ((now_y > next1_y and next2_y > next3_y) or (now_y < next1_y and next2_y < next3_y)) \
            and abs(next2_x - next1_x) <= MAX_DISTANCE:
        return "LEFT_RIGHT"
    else:
        return False

      
def compute_new_coord(now_coord, next3_coord_list, stair_type):
    '''Reduce coordinates to '-' shapes'''
    next1_coord, next2_coord, next3_coord = next3_coord_list[0], next3_coord_list[1], next3_coord_list[2]
    line_len1 = compute_line_len(now_coord, next1_coord, stair_type)
    line_len2 = compute_line_len(next2_coord, next3_coord, stair_type)

    now_x, now_y = get_xy(now_coord)
    next3_x, next3_y = get_xy(next3_coord)

    # compute the weighted average based on the length of the line
    new_coord1, new_coord2 = None, None
    if stair_type == "TOP_BOTTOM":
        new_y = weighted_average(now_y, next3_y, line_len1, line_len2)
        new_coord1 = (now_x, new_y)
        new_coord2 = (next3_x, new_y)
    else:  # stair_type == "LEFT_RIGHT"
        new_x = weighted_average(now_x, next3_x, line_len1, line_len2)
        new_coord1 = (new_x, now_y)
        new_coord2 = (new_x, next3_y)

    return new_coord1, new_coord2


#####################################
# main_logic
#####################################
def make_rectilinear(contour):
    '''rectilinearizate diagonal-case'''
    rectilinearized_contour = list()
    for idx in range(len(contour)):
        now_coord, prev_coord = contour[idx][0], contour[idx - 1][0]
        # add inner coord (prev -> "inner" -> now)
        if is_diagonal(now_coord, prev_coord):
            new_coord_list = get_rectilinear_coord(now_coord, prev_coord)
            for new_coord in new_coord_list:
                rectilinearized_contour.append([new_coord])
        # add now_coord
        rectilinearized_contour.append([tuple([now_coord[0], now_coord[1]])])

    # delete abnormal case: ex) (10, 21), "(10, 21)", (10, 30), "(22, 30)", (33, 30) => (10, 21), (10, 30), (33, 30)
    remove_unnecessary_points(rectilinearized_contour)
    return rectilinearized_contour

  
def flatten_stair_shape(rectilinearized_contour, MAX_DISTANCE, MAX_ITERATION):
    '''Refine Stair-case: Find coordinates consisting of 's' shapes & Reduce coordinates to '-' shape'''
    if len(rectilinearized_contour) >= 4:  # min_nums_of_points
        idx = 0
        iteration_count = 1
        stair_case_count = 0
        while iteration_count <= MAX_ITERATION:
            now_coord, next3_coord_list, next3_coord_idxs = get_now_next_3_coord_and_idxs(rectilinearized_contour, idx)
            stair_type = find_stair_shape(now_coord, next3_coord_list, MAX_DISTANCE)
            if stair_type:
                stair_case_count += 1
                new_coord1, new_coord2 = compute_new_coord(now_coord, next3_coord_list, stair_type)
                # update coordinates (Important!! After replace coordinates, remove items in order from the back of the list)
                rectilinearized_contour[idx][0] = new_coord1
                rectilinearized_contour[next3_coord_idxs[2]][0] = new_coord2
                del rectilinearized_contour[next3_coord_idxs[1]]
                if next3_coord_idxs[1] < next3_coord_idxs[0]:
                    del rectilinearized_contour[next3_coord_idxs[0]-1]
                else:
                    del rectilinearized_contour[next3_coord_idxs[0]]

                # delete abnormal case: ex) (10, 21), "(10, 21)", (10, 30), "(22, 30)", (33, 30) => (10, 21), (10, 30), (33, 30)
                # TODO(jiwon) : check sub list -> speed performance up

                less_than_idx = sum(tmp_idx < idx for tmp_idx in next3_coord_idxs[0:2])
                del_count = remove_unnecessary_points_range(rectilinearized_contour, idx - less_than_idx - 1)
                if del_count == 0:
                    # optional process (to produce more uniform output)
                    idx = idx + 1

                # end while 1
                if len(rectilinearized_contour) < 4:
                    break
            else:
                idx = idx + 1

            # end while 2
            if idx >= len(rectilinearized_contour):
                if stair_case_count == 0:
                    break
                else:
                    idx = 0
                    iteration_count = iteration_count + 1
                    stair_case_count = 0
            else:
                continue

    return rectilinearized_contour


#####################################
# Top-level logic (calls main_logic)
#####################################
def refine_image(pred_img_path, configs):
    '''
        cv2.findContours() -> make_rectilinear() -> flatten_stair_shape()
        * make rectilinear = convert diagonal to stair_shape
    '''
    KERNEL_SIZE = 3

    # prepare & preprocess(with morphology) mask_pred_img
    pred_img = read_img(pred_img_path)
    gray_pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2GRAY)
    _, gray_pred_img = cv2.threshold(gray_pred_img, 1, 255, cv2.THRESH_BINARY)
    gray_pred_img = cv2.morphologyEx(gray_pred_img, cv2.MORPH_CLOSE, np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8))

    # [Main_logic 1] find contours
    try:
        _, contours, hierarchy = cv2.findContours(gray_pred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, hierarchy = cv2.findContours(gray_pred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    del gray_pred_img

    # refinement process for each contour
    result_contours = list()
    if len(contours) == 0:
        return contours
    else:
        for idx_contour, contour in enumerate(contours):

            # [Main_logic 2] make rectilinear
            rectilinearized_contour = make_rectilinear(contour)

            # [Main_logic 3] flatten stair-shape
            result_contour = flatten_stair_shape(rectilinearized_contour, MAX_DISTANCE=configs['MAX_DISTANCE'], MAX_ITERATION=configs['MAX_ITERATION'])
            result_contours.append(np.asarray(result_contour))

        return result_contours


def check_small_contour(cnt, min_area=0, min_l=0):
    x,y,w,h = cv2.boundingRect(cnt)
    if (w<=min_l) or (h<=min_l) or (w*h<=min_area):
        return True
    else:
        return False


def get_polygon_list(cnts, hierarchy):
    result_children = list()
    result_contours = list()
    groups = dict()

    for idx, hier in enumerate(hierarchy[0]):
        """
        hier[0]: next (same level)
        hier[1]: previous (same level)
        hier[2]: first child
        hier[3]: parent
        """
        if hier[3]!=-1:
            # child contour
            try:
                groups[hier[3]].append(idx)
            except:
                groups[hier[3]] = [idx]

    for idx,cnt in enumerate(cnts):
        if hierarchy[0][idx][3]==-1:
            # not a child, draw!
            result_contours.append(cnt)
            child_list = list()
            if idx in groups.keys():
                # has children
                for child_idx in groups[idx]:
                    # append child
                    child_list.append(cnts[child_idx])
            result_children.append(child_list)

    return result_contours, result_children


def refine_image_direct(gray_pred_img, max_distance=5, max_iteration=30):
    '''
        cv2.findContours() -> make_rectilinear() -> flatten_stair_shape()
        * make rectilinear = convert diagonal to stair_shape
        without time check
    '''
    KERNEL_SIZE = 3
    
    # prepare & preprocess(with morphology) mask_pred_img
    try:
        gray_pred_img = cv2.morphologyEx(gray_pred_img, cv2.MORPH_CLOSE, np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8))
    except:
        print ("[WARNING] Large Contour")
        pass

    # [Main_logic 1] find contours
    try:
        _, contours, hierarchy = cv2.findContours(gray_pred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, hierarchy = cv2.findContours(gray_pred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    del gray_pred_img

    # refinement process for each contour
    result_contours = list()

    if len(contours) == 0:
        return [], []
    else:
        for idx_contour, contour in enumerate(contours):
            
            # [Main_logic 2] make rectilinear
            rectilinearized_contour = make_rectilinear(contour)
            # [Main_logic 3] flatten stair-shape
            result_contour = flatten_stair_shape(rectilinearized_contour, MAX_DISTANCE=max_distance, MAX_ITERATION=max_iteration)
            result_contours.append(np.asarray(result_contour))

    # get a pair of parent and children
    result_contours, result_children = get_polygon_list(result_contours, hierarchy)

    return result_contours, result_children


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input directory', default='input')
    parser.add_argument('--output', type=str, help='output directory', default='output')
    parser.add_argument('--smoothness', type=int, help='max line segment length to flatten', default=8)
    parser.add_argument('--iteration', type=int, help='num iterations of flattening', default=2)

    args = parser.parse_args()

    #####################################
    # Configuration
    #####################################
    configs = dict()
    configs['MAX_DISTANCE'] = args.smoothness   # Larger value : Increase smooth effects, Smaller value: More keep original shapes
    configs['MAX_ITERATION'] = args.iteration  # over 2

    #####################################
    # Refie prediction result
    #####################################
    input_dir = args.input
    output_dir = args.output

    samples = os.listdir(input_dir)

    for sample_file_name in samples:
        sample_file_path = os.path.join(input_dir, sample_file_name).replace("\\", "/")

        print(sample_file_path)
        # Refinement process
        result_contours = refine_image(sample_file_path, configs)
        result_image_contours = np.asarray(result_contours)

        # Get result image
        mask_result_img = cv2.imread(sample_file_path)
        mask_result_img = np.zeros(mask_result_img.shape)
        cv2.drawContours(mask_result_img, result_image_contours, -1, (255, 255, 255), -1)
    
        # Save result image
        cv2.imwrite(os.path.join(output_dir, sample_file_name).replace("\\", "/"), mask_result_img)
