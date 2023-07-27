
import process_receipt as pr

def test_many_receipts():
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
                rec = pr.extract_receipt(filename, 
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
    test_many_receipts()