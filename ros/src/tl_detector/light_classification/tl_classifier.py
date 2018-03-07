from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import cv2
import os 
import keras

cwd = os.path.dirname(os.path.realpath(__file__))

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        os.chdir(cwd)
        # load model for classification of traffic light color
        self.classifier = keras.models.load_model('model_.h5')
        self.cls_graph = tf.get_default_graph()

        # load SSD MobileNet petrained on COCO
        self.detect_graph = tf.Graph()        
        path_frozen_model = 'frozen_inference_graph.pb'

        with self.detect_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_frozen_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        self.sess = tf.Session(graph=self.detect_graph)

        self.image_tensor = self.detect_graph.get_tensor_by_name('image_tensor:0')
        self.detect_boxes = self.detect_graph.get_tensor_by_name('detection_boxes:0')
        self.detect_classes = self.detect_graph.get_tensor_by_name('detection_classes:0')
        self.detect_scores = self.detect_graph.get_tensor_by_name('detection_scores:0')

        self.img_height = 0
        self.img_width = 0

    def detect_traffic_lights(self, image):
	
	image = np.expand_dims(image, 0)
        with tf.Session(graph=self.detect_graph) as sess:
            (boxes, scores, classes) = sess.run([self.detect_boxes, self.detect_scores,\
                                                 self.detect_classes], feed_dict={self.image_tensor: image})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        return boxes, scores, classes

    def get_best_detected_traffic_light(self, boxes, scores, classes):
        inds = ((classes == 10) & (scores >= 0.15)).nonzero()
        box = [0,0,0,0]

        if len(inds[0])>0:
            top_score_ind = np.argmax(scores[inds])
        
            box[0] = int(boxes[top_score_ind,0]*self.img_height)
            box[2] = int(boxes[top_score_ind,2]*self.img_height)

            box[1] = int(boxes[top_score_ind,1]*self.img_width)
            box[3] = int(boxes[top_score_ind,3]*self.img_width)

        return box

    def crop_image(self, image, box):
        return image[box[0]:box[2],box[1]:box[3]]

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        self.img_height = image.shape[0]
        self.img_width = image.shape[1]
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = self.detect_traffic_lights(image)
        box = self.get_best_detected_traffic_light(boxes, scores, classes)
        if np.any(box):
            
            tl_image = self.crop_image(image, box)
	    #save_img = cv2.cvtColor(tl_image, cv2.COLOR_RGB2BGR)
            #cv2.imwrite('_cropped.jpg',save_img)
            tl_resized_image = cv2.resize(tl_image,(32,32))

            tl_resized_image = np.expand_dims(tl_resized_image, axis=0)

            with self.cls_graph.as_default():
                pred = self.classifier.predict(tl_resized_image)
                pred = np.argmax(pred)
                if pred == 0:
                    return TrafficLight.RED

                elif pred == 1:
                    return TrafficLight.YELLOW

                elif pred == 2:
                    return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
