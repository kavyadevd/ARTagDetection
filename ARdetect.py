
import numpy as np
import cv2
import scipy.fftpack as fp
from matplotlib import pyplot as plt
from copy import deepcopy


video = "1tagvideo.mp4"


# create video capture object
video_ = cv2.VideoCapture(video)


def EdgeDetect(frame):
    # To obtain cleaner segmentation
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    # Convert image to bw by thresholding
    #cv2.threshold(src, thresholdValue, maxVal, thresholdingTechnique)
    (thresh, im_bw) = cv2.threshold(frame, 140,
                                    255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    frame = im_bw

    cv2.imwrite('BeforeFFT.png', frame)

    # FFT
    F1 = fp.fft2((frame).astype(float))
    F2 = fp.fftshift(F1)

    # Create High pass filter
    (frame_width, frame_height) = frame.shape
    #print(frame_width,' ',frame_height)
    mid_width, mid_height = int(frame_width/2), int(frame_height/2)
    n = 25
    # select all but the first 50x50 (low) frequencies
    F2[mid_width-n:mid_width+n+1, mid_height-n:mid_height+n+1] = 0

    # inverse
    image1 = fp.ifft2(fp.ifftshift(F2)).real
    masked = cv2.bitwise_and(image1, image1, mask=frame)
    image1 = masked
    #cv2.imshow('After FFT',image1)
    cv2.imwrite('AfterFFT.png', image1)

    (thresh, im_bw) = cv2.threshold(frame, 110, 255, cv2.THRESH_BINARY)
    frame = im_bw
    return frame


def GetCorners(frame, color=255):
    white_pixels = np.array(np.where(frame == color))
    if color != 255:
        white_pixels = np.array(np.where(frame == color))
    arrx = white_pixels[0]
    arry = white_pixels[1]

    x1 = min(arrx)
    y1 = arry[(np.where(arrx == x1))[0][0]]
    x2 = max(arrx)
    y2 = arry[(np.where(arrx == x2))[0][0]]

    y3 = min(arry)
    x3 = arrx[(np.where(arry == y3))[0][0]]
    y4 = max(arry)
    x4 = arrx[(np.where(arry == y4))[0][0]]
    # if(color==255):
    #     x1+=10
    #     x4+=10
    #     y1-=10
    #     y4+=10

    #     x2-=10
    #     y2-=10
    #     x3-=10
    #     y3+=10

    # cv2.circle(original_image, (white_pixels[:,0][1], white_pixels[:,0][0]), 5, (0, 0, 255),-1)
    # cv2.circle(original_image, (white_pixels[:,-1][1], white_pixels[:,-1][0]), 5, (0, 255, 0,-1))

    corners = np.array([(y1, x1), (y3, x3), (y2, x2), (y4, x4)], np.int32)
    # print('corners',corners)
    return corners, x1, x2, x3, x4, y1, y2, y3, y4


def TagToGrid(img):
    grid = np.empty((8, 8))
    rows_ = int(img.shape[0] / 8)
    cols_ = int(img.shape[1] / 8)
    col_c = 0
    for i in range(0, img.shape[0], rows_):
        row_c = 0
        for j in range(0, img.shape[1], cols_):
            c_b = 0
            c_w = 0
            for x in range(0, rows_ - 1):
                for y in range(0, cols_ - 1):
                    if img[i + x][j + y] == 0:
                        c_b += 1
                    else:
                        c_w += 1
            if c_w >= c_b:
                grid[col_c][row_c] = 1
            else:
                grid[col_c][row_c] = 0
            row_c += 1
        col_c += 1
    return grid


def RemoveBG(original_image, corners):
    rect = cv2.boundingRect(corners)
    x, y, w, h = rect
    white_bg = 255*np.ones_like(original_image)
    white_bg[y:y+h, x:x+w] = original_image[y:y+h, x:x+w]

    stencil = np.zeros(original_image.shape[:-1]).astype(np.uint8)
    cv2.fillPoly(stencil, [corners], 255)
    sel = stencil != 255
    original_image[sel] = [255, 255, 255]
    original_image = cv2.GaussianBlur(original_image, (9, 9), 0)
    kernel = np.ones((5, 5), np.uint8)
    original_image = cv2.dilate(original_image, kernel, iterations=1)
    cv2.imwrite('Nobg.png', original_image)
    return original_image, w, h


def RotateTagBy(ar_tag):
    if ar_tag[2][2] == 0 and ar_tag[2][5] == 0 and ar_tag[5][2] == 0 and ar_tag[5][5] == 1:
        return 0, True
    elif ar_tag[2][2] == 0 and ar_tag[2][5] == 1 and ar_tag[5][2] == 0 and ar_tag[5][5] == 0:
        return 90, True
    elif ar_tag[2][2] == 0 and ar_tag[2][5] == 0 and ar_tag[5][2] == 1 and ar_tag[5][5] == 0:
        return -90, True
    elif ar_tag[2][2] == 1 and ar_tag[2][5] == 0 and ar_tag[5][2] == 0 and ar_tag[5][5] == 0:
        return 180, True
    return None, False


def GetContors(frame):
    #frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    pos_frame = video_.get(cv2.CAP_PROP_POS_FRAMES)
    # print(str(pos_frame)+" frames")
    original_image = frame
    if pos_frame == 9:
        cv2.imwrite(str(pos_frame)+".png", original_image)
    frame = cv2.GaussianBlur(frame, (9, 9), 0)
    if pos_frame == 9:
        plt.figure(figsize=(10, 10))
        plt.imshow(frame)
        plt.title('Captured frame')
        plt.show()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pos_frame = video_.get(cv2.CAP_PROP_POS_FRAMES)
    print(str(pos_frame)+" frames")
    frame = EdgeDetect(frame)
    if pos_frame == 9:
        plt.figure(figsize=(10, 10))
        plt.imshow(frame)
        plt.title('After FFT')
        plt.show()
    corners, x1, x2, x3, x4, y1, y2, y3, y4 = GetCorners(frame, 255)
    original_image, rec_w, rect_h = RemoveBG(original_image, corners)
    kernel = np.ones((10, 10), np.uint8)
    img_erosion = cv2.erode(original_image, kernel, iterations=1)
    img_erosion = cv2.dilate(img_erosion, kernel, iterations=1)
    cv2.imwrite('step1.png', original_image)
    frame = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    frame = EdgeDetect(frame)
    corners, x1, x2, x3, x4, y1, y2, y3, y4 = GetCorners(frame, 0)

    # Returns Output equivalent to inbuilt contor find function
    # arrc = [[[corners[0][0], corners[0][1], 'TL'], [corners[3][0], corners[3][1], 'TR'], [
    #     corners[2][0], corners[2][1], 'BR'], [corners[1][0], corners[1][1], 'BL'], ]]
    arrc_ = [0, 0, 0, 0]
    arrc_[0] = [corners[0][0], corners[0][1]]
    arrc_[1] = [corners[3][0], corners[3][1]]
    arrc_[2] = [corners[1][0], corners[1][1]]
    arrc_[3] = [corners[2][0], corners[2][1]]
    return arrc_, threshold


def DecodeTag(ar_tag):      # Decode Tag
    skew, found = RotateTagBy(ar_tag)
    if not found:
        return found, None
    else:
        if skew == 0:
            id = (ar_tag[3][3]) + (ar_tag[4][3] * 8) + \
                (ar_tag[4][4] * 4) + (ar_tag[3][4] * 2)
        elif skew == 90:
            id = (ar_tag[3][3] * 2) + (ar_tag[3][4] * 4) + \
                (ar_tag[4][4] * 8) + (ar_tag[4][3])
        elif skew == 180:
            id = (ar_tag[3][3] * 4) + (ar_tag[4][3] * 2) + \
                (ar_tag[4][4]) + (ar_tag[3][4] * 8)
        elif skew == -90:
            id = (ar_tag[3][3] * 1) + (ar_tag[3][4]) + \
                (ar_tag[4][4] * 2) + (ar_tag[4][3] * 4)
        return found, id


def GetHInv(src, dest):
    index = 0
    M = np.empty((8, 9))
    for i in range(0, len(src)):
        x1 = src[i][0]
        y1 = src[i][1]
        x2 = dest[i][0]
        y2 = dest[i][1]
        M[index] = np.array([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        M[index + 1] = np.array([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
        index += 2
    _, __, V = np.linalg.svd(M, full_matrices=True)
    V = (deepcopy(V)) / (deepcopy(V[8][8]))
    H = V[8, :].reshape(3, 3)
    H_inv = np.linalg.inv(H)
    return H_inv


# wait till video is playing
while not video_.isOpened():
    print("Loading Video")
    video_ = cv2.VideoCapture("1tagvideo.mp4")
    cv2.waitKey(1000)
while video_.isOpened():
    TagID = 0
    active_, img_ar = video_.read()
    pos_frame = video_.get(cv2.CAP_PROP_POS_FRAMES)
    if active_:
        new_width = 960
        new_height = 540
        img_ar = cv2.resize(img_ar, (new_width, new_height),
                            interpolation=cv2.INTER_NEAREST)
        img_raw = img_ar.copy()
        cv2.putText(img_raw, 'Press esc to exit', (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 218, 20), 1, cv2.LINE_AA)
        # cv2.imwrite('raw.png',img_raw)
        cornerAR, threshold = GetContors(img_ar)
        if len(cornerAR) > 0 and (not (0 in cornerAR)):
            c = np.array([cornerAR[0], cornerAR[2], cornerAR[3], cornerAR[1]])
            img_raw = cv2.polylines(img_raw, [c], True, (0, 255, 0), 3)
            cv2.circle(
                img_raw, (cornerAR[0][0], cornerAR[0][1]), 8, (22, 244, 238), -1)
            cv2.circle(
                img_raw, (cornerAR[1][0], cornerAR[1][1]), 8, (57, 255, 20), -1)
            cv2.circle(
                img_raw, (cornerAR[2][0], cornerAR[2][1]), 8, (255, 218, 20), -1)
            cv2.circle(img_raw, (cornerAR[3][0],
                       cornerAR[3][1]), 8, (0, 0, 255), -1)

            HInv = GetHInv(cornerAR, [[0, 0], [0, 47], [47, 0], [47, 47]])
            tag = np.zeros((48, 48))

            # WarpPerspective
            for m in range(0, 48):
                for n in range(0, 48):
                    x1, y1, z1 = np.matmul(HInv, [m, n, 1])
                    if new_width > int(x1 / z1) > 0 and new_height > int(y1 / z1) > 0:
                        tag[m][n] = threshold[int(y1 / z1)][int(x1 / z1)]

            tag = TagToGrid(tag)
            flag, tag_id = DecodeTag(tag)
            if tag_id:
                TagID = tag_id
            if flag:
                text = 'Tag ID'+str(tag_id)
                cv2.putText(img_raw, text, (img_raw.shape[1]-100, img_raw.shape[0]-100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 218, 20), 1, cv2.LINE_AA)
        cv2.imshow('Frame', img_raw)
        if cv2.waitKey(1) & 0xFF == 27:
            print('Tag ID is: ', TagID)
            cv2.imwrite('Detected.png', img_raw)
            break
    else:
        break

video_.release()
cv2.destroyAllWindows()
