import Common as common
import Camera as cameraModule
import Inference as inferenceModule

if __name__ == "__main__": 
    # Load and prepare model
    model_file_path = 'Model\\mobilenet_v1_1.0_224_quant.tflite'
    labels_file_path = 'Model\\labels_mobilenet_quant_v1_224.txt'

    # Initialize model
    model = inferenceModule.Inference(model_file_path, labels_file_path)

    # Create camera capture
    camera = cameraModule.Camera()

    # Test on the image from a file
    image = common.opencv.imread('TestImages\\cat.jfif')
    image_label = model.label_image(image)
    
    # Display results
    camera.display_image_with_label(image, image_label)
    
    # Get camera image
    image = camera.capture_frame(True)
    label = model.label_image(image)

    # Display results
    camera.display_current_frame_with_label(label)    