<div>
  <p>This code is an implementation of a YOLOv7 detector along with a Realsense D435i camera, which is used for object detection in real-time. The main purpose is to detect objects and obtain data such as position, distance, and orientation.</p>
  <h2>Requirements</h2>
  <ul>
    <li>Python 3.7 or later</li>
    <li>Pytorch 1.7.0 or later</li>
    <li>OpenCV 4.1.2 or later</li>
    <li>pyrealsense2 2.50.0 or later</li>
    <li>scipy 1.4.1 or later</li>
    <li>numpy 1.18.5 or later</li>
  </ul>
  <h2>Usage</h2>
  <p>To run the code, execute the detect function from the main.py file. The function has the following input arguments:</p>
  <ul>
    <li><code>--weights</code> : <code>str</code> : model.pt path(s)</li>
    <li><code>--source</code> : <code>str</code> : source (file/folder, 0 for webcam)</li>
    <li><code>--img-size</code> : <code>int</code> : inference size (pixels)</li>
    <li><code>--conf-thres</code> : <code>float</code> : object confidence threshold</li>
    <li><code>--iou-thres</code> : <code>float</code> : IOU threshold for NMS</li>
    <li><code>--device</code> : <code>str</code> : cuda device, i.e. 0 or 0,1,2,3 or cpu</li>
  </ul>
  <p>Example usage:</p>
  <pre>
    <code>python main.py --weights yolov7s.pt --source 0 --img-size 640 --conf-thres 0.4 --iou-thres 0.5</code>
  </pre>
  <p>To stop the detection, press the <code>q</code> key.</p>
  <h2>Results</h2>
  <p>The code will output a video stream that will display the detected objects along with the corresponding bounding boxes, labels, and orientation angles. Additionally, the data will be saved in a .mat file, containing the object coordinates, distance, and orientation data at time relative to code initialization.</p>
  <h2>Acknowledgements</h2>
  <p>This code was implemented using the YOLOv7 repository by Ultralytics LLC.</p>
  <h2>License</h2>
  <p>This project is licensed under the MIT License.</p>
</div>
