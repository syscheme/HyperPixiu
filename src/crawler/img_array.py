from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.open('/mnt/d/test.jpg') # 本地一个文件
mode_list = ['1', 'L', 'I', 'F', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr' ]
mode_list = ['RGBA', 'RGB' ]
for mode in mode_list:
    img = image.convert(mode)
    img_data = np.array(img)
    print('img_{:>1}.shape: {}' .format(mode, img_data.shape))
    print('img_{:>}_data[0, 0]: {}'.format(mode, img_data[0, 0]))
    try: 
        img.save('/mnt/d/test_%s.bmp' % mode) 
    except:
        pass
    print('---')
