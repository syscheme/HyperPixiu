# the record sequence consists of
# - the first base image, precented once
#      . 5min-KLines x240 stack for a week, KLine[0] is the most recent, the rest 0-prefilled
#      . day-KLines x 220 stack for a year, KLine[0] is the most recnet, the rest 0-prefilled
# - the 2nd+ record is incremental data
#      . one new 5min-KLines at KLine[0] in addtion. the new record is expected to shift the previous record by 1
#      . one new day-Klines, shifting previous record is expected too if pushing a new Kline
# https://blog.csdn.net/zx520113/article/details/84556489

class RecorderSaver(symbol, folder, KLs_5min=240, KLs_day=220) :
    def openfile(self) :
        self._writer = tf.python_io.TFRecordWriter(save_file)
        self._irec =0
    
    def write(KL5min, K)

def Save_data(filename='..\\imgout',save_file="train.trec",img_size=(224,224)):
    """
    将文件夹下的图像数据存储在.tfrecords文件中,通过调用TensorFlow中的train.Example实现
    :param filename: 文件地址
    :param save_file: 保存的文件地址（文件名）
    :return: None
    """
    writer = tf.python_io.TFRecordWriter(save_file)
    for index in os.listdir(filename):
        print("Label",index)
        class_path = filename +"\\"+ index+"\\"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize(img_size)
            img = img.tobytes() #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(index)])),
                    'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))}))
            writer.write(example.SerializeToString())
    writer.close()
