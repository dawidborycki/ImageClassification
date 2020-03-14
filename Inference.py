from tensorflow import lite as tflite
import numpy as np
from Common import opencv

class Inference(object):    
    
    def __init__(self, model_file_path, labels_file_path):
        # Load model
        self.load_model_and_configure(model_file_path)

        # Load labels
        self.load_labels_from_file(labels_file_path)
    
    def load_model_and_configure(self, model_path):
        """ Loads the model and configures input image dimensions accordingly
        : model_path: A full path to the file containing the model
        """
        # Load model from file
        self.interpreter = tflite.Interpreter(model_path)

        # Allocate tensors
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Store input image dimensions
        self.input_image_height = self.input_details[0]['shape'][1]
        self.input_image_width = self.input_details[0]['shape'][2]

    def load_labels_from_file(self, file_path):
        """ Loads image labels from the text file
        : file_path: A full path to the text file, containing image labels
        """
        with open(file_path, 'r') as file:
            self.labels = [line.strip() for line in file.readlines()]

    def prepare_image(self, image):
        """ Prepares image for the TensorFlow inference
        : image: An input image
        """
        # Convert image to BGR
        image = opencv.cvtColor(image, opencv.COLOR_BGR2RGB)        

        # Get new size
        new_size = (self.input_image_height, self.input_image_width);

        # Resize
        image = opencv.resize(image, new_size, interpolation = opencv.INTER_AREA) 

        return image

    def label_image(self, image):
        """ Labels an image
        : image: An input image to be labeled
        """

        # Prepare image
        image = self.prepare_image(image)

        # Add dummy dimension
        input_data = np.expand_dims(image, axis=0)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get result
        inference_result = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Remove dummy dimension
        inference_result = np.squeeze(inference_result)

        # Obtain the label with the highest score
        top_one = inference_result.argmax()
        return self.labels[top_one]