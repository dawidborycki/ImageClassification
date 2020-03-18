import Common as common
from Common import opencv

class Camera(object):    

    def __init__(self):
        # Initialize the camera capture
        self.camera_capture = opencv.VideoCapture(0)
        
    def capture_frame(self, ignore_first_frame):
        # Get frame, ignore the first one if needed
        if(ignore_first_frame):
            self.camera_capture.read()
            
        (capture_status, self.current_camera_frame) = self.camera_capture.read()

        # Verify capture status
        if(capture_status):
            return self.current_camera_frame

        else:
            # Print error to the console
            print(common.CAPTURE_FAILED)

    def display_image_with_label(self, image, label):
        # Put label on the image
        image_with_label = opencv.putText(image, label, 
                                          common.TEXT_ORIGIN, 
                                          common.FONT_FACE, 
                                          common.FONT_SCALE, 
                                          common.GREEN,
                                          common.FONT_THICKNESS,
                                          common.FONT_LINE)

        # Display image
        opencv.imshow(common.PREVIEW_WINDOW_NAME, image_with_label)
            
        # Wait until user presses any key
        opencv.waitKey()  

    def display_current_frame_with_label(self, label):
        self.display_image_with_label(self.current_camera_frame, label)