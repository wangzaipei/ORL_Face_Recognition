from PIL import Image
import os, glob

def batch_image(in_dir, out_dir):
    #判断输入输出路径是否存在
    if not os.path.exists(out_dir):
        print(out_dir, 'is not existed.')
        os.mkdir(out_dir)
    if not os.path.exists(in_dir):
        print(in_dir, 'is not existed.')
        return -1
    #遍历路径的图片文件
    count = 1
    for files in glob.glob(in_dir + '/*'):
        filepath, filename = os.path.split(files)
        #分割文件名以及格式名，在文件名后面添加jpg格式名
        name=filename[0:9].split('.')
        # print(name)
        out_file = name[0] + '.jpg'
        # print(filepath,',',filename, ',', out_file)
        #输出图片到输出路径
        im = Image.open(files)
        new_path = os.path.join(out_dir, out_file)
        print(count, ',', new_path)
        count = count + 1
        im.save(os.path.join(out_dir, out_file))


if __name__ == '__main__':
    #40个人脸，循环调用
    for i in range(1,41):
        batch_image('s'+str(i), '../data/s'+str(i))
