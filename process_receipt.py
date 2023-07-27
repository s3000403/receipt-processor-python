# import the necessary packages
import numpy as np
from imutils.perspective import four_point_transform
import imutils
#import pytesseract
from helpers import open_image_cv
from PIL import Image
import cv2

# Based on following site: https://pyimagesearch.com/2021/10/27/automatically-ocring-receipts-and-scans/

def identify_receipt_bounds(orig, 
                            blur_size=5, 
                            canny_min_thresh=75, 
                            canny_max_thresh=200,
                            resize_width = 500,
                            ):
    # Resize image, and compute the ratio of the *new* width to the 
    # *old* width
    image = orig.copy()
    image = imutils.resize(image, width=resize_width)
    ratio = orig.shape[1] / float(image.shape[1])

    # convert the image to grayscale, blur it slightly, and then apply
    # edge detection
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size,), 0)
    edged   = cv2.Canny(blurred, canny_min_thresh, canny_max_thresh)

    # find contours in the edge map and sort them by size in descending
    # order
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the receipt outline
    receiptCnt = np.ndarray((0,1,2))
    # loop over the contours (from largest to smallest since we sorted)
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the receipt
        x,y,w,h = cv2.boundingRect(c)
        # if this contour is the largest 4 sided polygon use this as the receipt
        if len(approx) == 4:
            receiptCnt = approx
            break
    # if we failed to find a valid contour return the empty contour
    if len(receiptCnt)==0:
        return receiptCnt
    # else reshape the contour into bounds (scaled back to the original image size) 
    # and return these bounds
    receiptBounds = receiptCnt.reshape(4, 2) * ratio
    return receiptBounds
    
def extract_receipt(filename, 
                    blur_size=5, 
                    canny_min_thresh=75, 
                    canny_max_thresh=200,
                    resize_width = 500,
                    ):
    debug = 1

    # load the input image from disk
    orig = open_image_cv(filename)

    receiptBounds = identify_receipt_bounds(orig, 
                    blur_size=blur_size, 
                    canny_min_thresh=canny_min_thresh, 
                    canny_max_thresh=canny_max_thresh,
                    resize_width = resize_width,
                    )
    
    # if the receipt contour is empty then our script could not find the
    # outline and we should be notified
    if len(receiptBounds)==0:
        return np.ndarray((0,0,3))

    # apply a four-point perspective transform to the *original* image to
    # obtain a top-down bird's-eye view of the receipt
    receipt = four_point_transform(orig, receiptBounds)
    #print(receipt.shape)
    
    # check to see if we should draw the contour of the receipt on the
    # image and then display it to our screen
    if debug > 0:
        print('Plotting debug plots')
        output = orig.copy()
        receiptCnt = receiptBounds.reshape(4,1,2).round().astype(int)
        cv2.drawContours(output, [receiptCnt], -1, (0, 255, 0), 10)
        cv2.imshow("Receipt Outline", imutils.resize(output, width=resize_width))
        cv2.imshow("Receipt Transform", imutils.resize(receipt, width=resize_width))
        
        cv2.waitKey()
        cv2.destroyAllWindows()
    return receipt

if __name__=='__main__':
    filename = 'data2\\220728.HEIC'
    #filename = 'data2\\220925.HEIC'#fail
    filename = 'data2\\221018.HEIC'#fail
    #filename = 'data2\\221103.HEIC'#fail
    #filename = 'data2\\221118.HEIC'
    #filename = 'data2\\221209.HEIC'
    #filename = 'data2\\230517.HEIC'
    #filename = 'data2\\230619.HEIC'
    #filename = 'data2\\230702.HEIC'
    #filename = 'data2\\230717.HEIC'
    receipt = extract_receipt(filename, blur_size=5, resize_width=300)
    #cv2.imshow("Receipt", imutils.resize(receipt, width=300))
    #cv2.waitKey()
    #cv2.destroyAllWindows()