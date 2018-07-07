# 批量重命名文件
import os

class ImageRename():
    def __init__(self,path):
        #self.path = 'D:\\DATASETS\\PCBA'
        self.path = path
    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        i = 0

        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.JPG'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), '2018-0000' + format(str(i), '0>3s') + '.jpg')
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i = i + 1
            elif item.endswith('.png') or item.endswith('.PNG'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), '2018-0000' + format(str(i), '0>3s') + '.png')
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i = i + 1

        print('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    path = input('please input path:')
    newname = ImageRename(path)
    newname.rename()
