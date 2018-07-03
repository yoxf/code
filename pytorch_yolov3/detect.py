import sys
import time

from PIL import Image
import utils
from darknet import Darknet

namesfile=None
def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    
    start = time.time()
    boxes = utils.do_detect(m, sized, 0.5, 0.4)
    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = utils.load_class_names(namesfile)
    utils.plot_boxes(img, boxes, 'predictions.jpg', class_names)

if __name__ == '__main__':
   
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        globals()["namesfile"] = sys.argv[4]
        detect(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile names')
