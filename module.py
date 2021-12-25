import cv2, logging, argparse
import numpy as np

parser = argparse.ArgumentParser("poses")
parser.add_argument('-i',"--input", type = str, required = True, default = False, help = "path image ...")
logging.basicConfig(filename=f'log/app.log', filemode='w', format='%(asctime)s - %(message)s', level = logging.INFO, datefmt='%d-%b-%y %H:%M:%S')



class PoseDnn(object):
    def __init__(self, path_weights, path_classes):
        self.net = cv2.dnn.readNet(path_weights)
        self.class_names = self.read_classes(path_classes)
        self.layer = 'StatefulPartitionedCall/StatefulPartitionedCall/model/cls/Softmax'
        

    def read_classes(self, path):             
        with open(f'{path}', 'r') as f:
            class_labels = f.readlines()
        class_labels = [cls.strip() for cls in class_labels]
        return class_labels    
    
    def __call__(self, input_image:str):
        image = cv2.imread(input_image)
        img_blob = cv2.dnn.blobFromImage(image, 1/255., (224,224), swapRB=True, crop=False)
        self.net.setInput(img_blob)
        output = self.net.forward(self.layer)
        pred_cls = np.argmax(output)
        classNames = self.class_names[pred_cls]
        return classNames


if __name__ == '__main__':
    args = parser.parse_args()
    path_weights = 'weights/frozen.pb'
    path_classes = 'weights/classes.txt'
    net = PoseDnn(path_weights, path_classes)
    classes = net(args.input)
    logging.info(f'classes name: {classes}')
    
    
