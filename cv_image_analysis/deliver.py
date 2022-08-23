import datetime
import os
import numpy as np
import cv2

colors = {"1.jpg": (0, 255, 0),
          "2.jpg": (255, 0, 0),
          "3.jpg": (221,160,221),
          "4.jpg": (0, 0, 255),
          "5.jpg": (0, 165, 255)}

template_threshold = {"1.jpg": 0.7,
                      "2.jpg": 0.72,
                      "3.jpg": 0.7,
                      "4.jpg": 0.75,
                      "5.jpg": 0.87}



def create_templates(img, contours):
    for idx in range(len(contours)):
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contours, idx, 255, -1)
        out = np.zeros_like(img)
        out[mask == 255] = img[mask == 255]

        (y, x, c) = np.where(mask == 255)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        out = img.copy()[topy:bottomy + 1, topx:bottomx + 1, :]

        image = cv2.rectangle(img.copy(), (topx, topy), (bottomx, bottomy), (0, 255, 255), 2)
        cv2.imshow("original", image)
        cv2.imshow('Output', out)
        key = cv2.waitKey(0)
        if key == ord('y'):
            cv2.imwrite(f"templates/{round(datetime.datetime.utcnow().timestamp())}_{idx}.jpg", out)
        elif key == ord('q'):
            break



def bad_match(image_path):
    image = cv2.imread(image_path)
    fixed_image = fix_image(image, False)
    imgray = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    new_cnts = []
    for cnt in cntsSorted:
        if cv2.contourArea(cnt) < 4000 and cv2.contourArea(cnt) > 2500:
            # print(cv2.contourArea(cnt))
            new_cnts.append(cnt)


    create_templates(fixed_image, new_cnts)
    if new_cnts:
        return 1
    else:
        fix_image(image, True)
        cv2.destroyAllWindows()

def create_matrix(points_list):
    min_x = min([i[0][0] for i in points_list])
    min_y = min([i[0][1] for i in points_list])

    points_list = [((round((i[0][0] - min_x) / 60), round((i[0][1] - min_y) / 60)), i[1]) for i in points_list]
    points_list.sort(key=lambda x: x[0][0])
    points_list.sort(key=lambda x: x[0][1])

    last_point = 0,0
    all_lines = []
    current_line = []
    new_line = False
    for pt, name in points_list:
        # print(pt, name)
        if pt[1] > last_point[1]:
            new_line = True
        last_point = pt

        if new_line:
            all_lines.append(current_line)
            current_line = []
            new_line = False

        current_line.append((pt, name))

    all_lines.append(current_line)

    matrix = np.zeros((4, 4))
    for line in all_lines:
        for point, name in line:
            matrix[point[1]][point[0]] = int(name)

    return matrix


def get_matches(img, template, threshold):
    pt_over_threshold = []
    w, h = template.shape[0], template.shape[1]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        precision = (res[pt[1]][pt[0]])
        pt_over_threshold.append((pt, w, h, precision))

    return pt_over_threshold


def filter_detections(list_of_points):
    outpoints = []
    intersection_percent = 0.2

    threshold_of_drawen_pts = []
    list_of_points.sort(key=lambda x: x[3], reverse=True)
    for pt1, w1, h1, precision1 in (list_of_points):
        add_new_point = True
        R1 = (pt1[0], pt1[1], pt1[0]+w1, pt1[1]+h1)
        for pt2, w2, h2 in outpoints:
            intersection_allowed = intersection_percent*w2/2 + intersection_percent*w1/2
            R2 = (pt2[0], pt2[1], pt2[0] + w2, pt2[1] + h2)
            if not ((R1[0]+intersection_allowed >= R2[2]) or (R1[2] <= R2[0]+intersection_allowed) or (R1[3] <= R2[1]+intersection_allowed) or (R1[1]+intersection_allowed >= R2[3])):
                add_new_point = False
                break
        if add_new_point:
            outpoints.append((pt1, w1, h1))
            threshold_of_drawen_pts.append(precision1)

    return outpoints


def fix_image(image, show):
    img_blur = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    img_blur = cv2.GaussianBlur(img_blur, (5, 5), cv2.BORDER_DEFAULT)
    img_blur = cv2.GaussianBlur(img_blur, (5, 5), cv2.BORDER_DEFAULT)
    img_blur = cv2.GaussianBlur(img_blur, (5, 5), cv2.BORDER_DEFAULT)

    imgray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    new_contours = []
    for cnt in cntsSorted:
        if cv2.contourArea(cnt) < image.shape[0] * image.shape[1] * 0.6:
            new_contours.append(cnt)

    peri = cv2.arcLength(new_contours[0], True)
    corners = cv2.approxPolyDP(new_contours[0], 0.04 * peri, True)
    corners = [corners[i][0] for i in range(corners.shape[0])]

    pts1 = np.float32([corners[1], corners[0], corners[2], corners[3]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (300, 300))

    if show:
        cv2.imshow('img', image)
        cv2.imshow("warped", dst)
        cv2.imshow("edge", thresh)

        image_contours = cv2.drawContours(image.copy(), new_contours, 0, (0, 255, 0), 3)
        cv2.imshow('image_contours', image_contours)

        # cv2.imwrite("templates/demo.jpg", dst)
        cv2.waitKey(0)
    return dst


def main(image_path, display=False):
    os.makedirs("output_images", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    image = cv2.imread(image_path)
    fixed_image = fix_image(image, display)
    all_points = {}
    all_detections = []
    color_map = {}
    for i in os.listdir("templates"):
        template = cv2.imread(f"templates/{i}")
        i = f"{i[0]}.jpg"
        image = fixed_image.copy()
        output_points = get_matches(image, template, template_threshold[i])
        for op in output_points:
            all_detections.append(op)
            color_map.update({op[0]: colors[i]})
            all_points.update({op[0]: i[0]})

    filtered_points = filter_detections(all_detections)

    filterd_pts = []
    for pt, w, h in filtered_points:
        img_with_rects = cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), color_map[pt], -1)
        filterd_pts.append(pt)

        # all_points[i].append(pt)
    cv2.destroyAllWindows()
    image = fixed_image.copy()

    if filtered_points:

        matrix_list = []
        for op in all_points.keys():
            if op in filterd_pts:
                matrix_list.append((op, all_points[op]))
        output_matrix = create_matrix(matrix_list)
        
        if display:
            cv2.imshow("final", image)
            cv2.imshow("mask", img_with_rects)
            # cv2.imwrite(f"output_images/{image_name.split('.')[0]}_final_image.jpg", image)
            # cv2.imwrite(f"output_images/{image_name.split('.')[0]}_mask_image.jpg", img_with_rects)
            print(output_matrix)
            cv2.waitKey(0)

        return output_matrix


if __name__ == '__main__':
    os.makedirs("test_images", exist_ok=True)
    for image_name in os.listdir("test_images"):
        card_matrix = main(f"test_images/{image_name}")
        if card_matrix is None:
            print(f"couldnt handle image {image_name}")
            print("\n")
        else:
            print(f"{image_name}:")
            print(card_matrix)
            print("\n")
