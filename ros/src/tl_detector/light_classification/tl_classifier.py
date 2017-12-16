from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):

        # Default value
        self.current_light = TrafficLight.UNKNOWN

        #load model
        model_labels = 'light_classification/model/retrained_labels.txt'
        model_graph = 'light_classification/model/retrained_graph.pb'

        #load labels
        self.labels = self.load_labels(model_labels)

        #load graph and create session
        graph = self.load_graph(model_graph)

        with tf.Session(graph=graph) as sess:
            # Define input & output layer
            self.layers(graph)
            # Save session
            self.sess = sess

    # Reference Repo: tensorflow-for-poets-2
    # https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/scripts/label_image.py
    #Load labels
    def load_labels(self,label_file):
        labels = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            labels.append(l.rstrip())
        return labels

    #Load graph
    def load_graph(self, model_graph):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(model_graph, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    # Define input & output layer
    def layers(self, graph):
        input_layer = "input"
        output_layer = "final_result"
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        self.input_operation = graph.get_operation_by_name(input_name)
        self.output_operation = graph.get_operation_by_name(output_name)

    #Pipeline to prepare Image to the model mobilenet_1.0_224":
    #Image pipeline : raw image(sim/real) >> resize >> normalize >> reshape to model
    def image_pipeline(self, img):
        #ARCHITECTURE = "mobilenet_1.0_224"
        image_data = cv2.resize(img, (224,224))
        #normalize image
        image_data = (image_data - 128.)/128.
        #reshape to model input
        image_data = np.reshape(image_data, (1, 224,224,3))
        return image_data


    #Run session with preloaded graph & labels
    def sess_run(self, image_data, labels):
        #Get tensor and get predictions
        results = self.sess.run(self.output_operation.outputs[0],
                                {self.input_operation.outputs[0]: image_data})
        results = np.squeeze(results)

        # Sort to show labels in order of confidence
        top_k = results.argsort()[-1:][::-1]
        for k in top_k:
            label = labels[k]
            score = results[k]
            #Test
            print('rob >> label={} score={:04.2f}'.format(label, score))
        return label

    #Predict label
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #implement light color prediction
        image_data = self.image_pipeline(image)
        predict_label = self.sess_run(image_data, self.labels)

        if predict_label == 'red':
            self.current_light = TrafficLight.RED
        else:
            self.current_light = TrafficLight.UNKNOWN
        #Test
        print('rob >> TrafficLight:{}'.format(self.current_light))
        return self.current_light
