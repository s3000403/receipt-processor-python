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
    receiptCnt = np.array([])
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the receipt
        x,y,w,h = cv2.boundingRect(c)
        #print(len(approx), cv2.contourArea(c),w*h,w,h,(w*h-cv2.contourArea(c))/cv2.contourArea(c))
        if len(approx) == 4:
            receiptCnt = approx
            break
    if len(receiptCnt)==0:
        return receiptCnt
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
    print(len(receiptBounds))
    # if the receipt contour is empty then our script could not find the
    # outline and we should be notified
    if len(receiptBounds)==0:
        return np.ndarray((0,0,3))

    # apply a four-point perspective transform to the *original* image to
    # obtain a top-down bird's-eye view of the receipt
    receipt = four_point_transform(orig, receiptBounds)
    
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

def test__extract_receipt():
    import pandas as pd
    from collections import OrderedDict
    from itertools import product
    import matplotlib.pyplot as plt
    test__filenames        = [
                                ('data2\\220728.HEIC',4505384),
                                ('data2\\220925.HEIC',4942470),
                                ('data2\\221018.HEIC',4605448),
                                ('data2\\221103.HEIC',5028205),
                                ('data2\\221118.HEIC',5303348),
                                ('data2\\221209.HEIC',5100687),
                                ('data2\\230517.HEIC',4486743),
                                ('data2\\230619.HEIC',4768295),
                                ('data2\\230702.HEIC',4549056),
                                ('data2\\230717.HEIC',4788268),
                              ]
    test__blur_size        = [5,7,9,11]
    test__canny_min_thresh = [ 75, 25, 50,100,125] #[ 75, 25, 50,100,125] np.linspace(25,200,15)
    test__canny_max_thresh = [200,150,175,225,250] #[200,150,175,225,250] np.linspace(100,300,15)
    test__resize_width     = [500,300,400,600]

    data = OrderedDict()

    for blur_size in test__blur_size:
        passFail = []
        print(blur_size)
        for (filename,thesh) in test__filenames:
#            sizes = []
            print(filename,end='')
            for (canny_minT,canny_maxT,resize_width) in product(
                                                    test__canny_min_thresh,
                                                    test__canny_max_thresh,
                                                    test__resize_width,
                                                    ):
                if canny_minT>canny_maxT:
                    print('o',end='')
                    continue
                print('x',end='')
                rec = extract_receipt(filename, 
                                    blur_size=blur_size, 
                                    canny_min_thresh=canny_minT, 
                                    canny_max_thresh=canny_maxT,
                                    resize_width=resize_width
                                    )
                h,w = rec.shape[:2]
#                if h*w>3000*1000:
#                    sizes.append(h*w)
                if h*w>=thesh:
                    success=1
                else:
                    success=0
                passFail.append(success)
                #print(rec.shape)
                data.setdefault('filename',[])
                data.setdefault('blur_size',[])
                data.setdefault('canny_minT',[])
                data.setdefault('canny_maxT',[])
                data.setdefault('resize_width',[])
                data.setdefault('area',[])
                data.setdefault('success',[])
                data['filename'].append(filename)
                data['blur_size'].append(blur_size)
                data['canny_minT'].append(canny_minT)
                data['canny_maxT'].append(canny_maxT)
                data['resize_width'].append(resize_width)
                data['area'].append(w*h)
                data['success'].append(success)
#            print(int(min(sizes)*0.95))
            print('')
        print(np.mean(passFail))
    data = pd.DataFrame(data)
    data.to_csv('results.csv')
    return 

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
    #print(receipt.shape)
    #cv2.imshow("Receipt", imutils.resize(receipt, width=300))
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    #test__extract_receipt()