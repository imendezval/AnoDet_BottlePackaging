import cv2 as cv
import numpy as np

def prep_img(img):

    mask = cv.imread("C:/data/git/repo/Bottle_AnoDet/imgs/bin_mask_opt.jpg")


    roi_opt = [525, 215, 150, 400] #150, 400]
    roi_bound = 30
    # variations less than roi_bound

    img_cropped = img[roi_opt[1]:roi_opt[1]+roi_opt[3], roi_opt[0]:roi_opt[0]+roi_opt[2]]
    masked_img = cv.bitwise_and(img_cropped, mask)

    return masked_img


img_name = "ex1"
img = cv.imread("C:/data/git/repo/Bottle_AnoDet/imgs/sampled/no_anomaly/fa_masked_0100.jpg")
masked_img = prep_img(img)
cv.imwrite("test_opt.jpg", masked_img)

#cv.imshow("Image", img)
#roi = cv.selectROI("Image", img, showCrosshair=True, fromCenter=False)
#cv.destroyAllWindows()
#print(roi)
#roi_first = [600, 210, 170, 410]