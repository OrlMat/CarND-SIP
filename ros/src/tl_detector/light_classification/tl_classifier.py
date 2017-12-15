from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #load classifier
        model_labels = 'light_classification/model/retrained_labels.txt'
        model_graph = 'light_classification/model/retrained_graph.pb'

        #load labels
        self.labels = self.load_labels(model_labels)

        #load graph
        with tf.Session() as sess:
            self.load_graph(model_graph)
            self.sess = sess

    #Ref repo: https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/scripts/label_image.py
    #Load Graph
    def load_graph(model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    #load labels
    def load_labels(label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    #Pipeline to prepare Image to the model mobilenet_1.0_224":
    # resize to model input >> normalize image >> reshape
    def scale_image(self, img):
        image_data = cv2.resize(img, (224,224))
        image_data = (image_data - 128.)/128.
        image_data = np.reshape(image_data, (1, 224,224,3))
        return image_data

    #Run session with preload graph & labels
    def sess_run(self, image_data, labels, input_layer_name, output_layer_name,
                  num_top_k=1):

        softmax_tensor = self.sess.graph.get_tensor_by_name(output_layer_name)
        predictions, = self.sess.run(softmax_tensor, {input_layer_name: image_data, 'Placeholder:0':1.0})

        # Sort to show labels in order of confidence
        top_k = predictions.argsort()[-num_top_k:][::-1]
        for k in top_k:
            label = labels[k]
            score = predictions[k]
            print('%s (score = %.3f)' % (label, score))
        return label

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN
