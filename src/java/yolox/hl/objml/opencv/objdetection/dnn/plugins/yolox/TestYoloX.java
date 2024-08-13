package hl.objml.opencv.objdetection.dnn.plugins.yolox;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import hl.opencv.util.OpenCvUtil;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class TestYoloX {

    private Net net;
    private Size inputSize = new Size(640, 640);
    private float confThreshold = 0.1f; // Confidence threshold
    private Scalar mean = new Scalar(0, 0, 0); // Mean subtraction values

    public TestYoloX(String modelWeights) {
        this.net = Dnn.readNetFromONNX(modelWeights);
    }

    public List<Rect2d> detect(String imagePath) {
        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            System.err.println("Failed to load image.");
            return null;
        }

        Mat blob = Dnn.blobFromImage(image, 1 / 255.0, inputSize, mean, true, false);
System.out.println("blob="+blob);        
        net.setInput(blob);
        Mat detections = net.forward();

        List<Rect2d> boxes = new ArrayList<>();
        
System.out.println("detection="+detections);

        for (int i = 0; i < detections.rows(); i++) {
            float confidence = (float) detections.get(0, i)[4];
            if (confidence > confThreshold) {
                float centerX = (float) detections.get(0, i)[0] * image.cols();
                float centerY = (float) detections.get(0, i)[1] * image.rows();
                float width = (float) detections.get(0, i)[2] * image.cols();
                float height = (float) detections.get(0, i)[3] * image.rows();

                float left = centerX - width / 2;
                float top = centerY - height / 2;

                Rect2d box = new Rect2d(left, top, width, height);
                boxes.add(box);

                // Draw the bounding box and confidence on the image
                Imgproc.rectangle(image, new Point(left, top), new Point(left + width, top + height), new Scalar(0, 255, 0), 2);
                Imgproc.putText(image, String.format("%.2f", confidence), new Point(left, top - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0));
            }
        }
        
        System.out.println("Total Detection = " + boxes.size());
        if(boxes.size()>0)
        {
	        // Save the output image with bounding boxes
	        String outputImagePath = imagePath.substring(0, imagePath.lastIndexOf(".")) + "_output.png";
	        Imgcodecs.imwrite(outputImagePath, image);
	        System.out.println("Output saved to: " + outputImagePath);
        }
        return boxes;
    }

    public static void main(String[] args) {
        OpenCvUtil.initOpenCV();

        File fileModel = new File("./src/java/yolox/hl/objml/opencv/objdetection/dnn/plugins/yolox/yolox_s.onnx");
        File fileImage = new File("./test/images/dog_bike_car.png");

        if (!fileModel.exists()) {
            System.err.println("Model file not found.");
            return;
        }

        if (!fileImage.exists()) {
            System.err.println("Image file not found.");
            return;
        }

        TestYoloX detector = new TestYoloX(fileModel.getAbsolutePath());
        detector.detect(fileImage.getAbsolutePath());
    }
}
