package hl.objml.opencv.objdetection.dnn.plugins.yolo;

import hl.opencv.util.OpenCvUtil;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class TestYoloX {
	

    public static void main(String[] args) {
    	
    	OpenCvUtil.initOpenCV();
    	
		File folderCur = new File(".");    	
    	String sCurPath = folderCur.getAbsolutePath();

    	List<File> listImages = new ArrayList<File>();
    	String sImagePath = sCurPath+"/test/images";
    	String[] sFileNames = new String[] {
    			"/world-largest-selfie.png", 
    			"/dog_bike_car.png", 
    			"/coco2017-id216861.png"};
 		for(String sFileName : sFileNames)
 		{
 			File f = new File(sImagePath+sFileName);
 			if(f.isFile())
 			{
 				listImages.add(f);
 			}
 		}
 		
		String sPackageFolder = TestYoloX.class.getPackageName().replaceAll("\\.", "\\/");
		File fileOnnxModel = new File(sCurPath+"/src/java/yolox/"+sPackageFolder+"/yolox_l.onnx");
		
		if(fileOnnxModel.isFile() && listImages.size()>0)
		{
			
	        // Load the model
	        Net net = Dnn.readNetFromONNX(fileOnnxModel.getAbsolutePath());
	        
	        
	        String sOutputFileFolder = sImagePath+"/output/"+System.currentTimeMillis()+"/";
	        new File(sOutputFileFolder).mkdirs();
	        
			for(File fileImage : listImages)
			{
			       // Load the input image
		        Mat image = Imgcodecs.imread(fileImage.getAbsolutePath());
	
		        // Pre-process the image: resize, normalize, etc.
		        Size inputSize = new Size(640, 640);
		        Mat blob = Dnn.blobFromImage(image, 1/255.0, inputSize, Scalar.all(0), true, false);
	
		        // Set input to the network
		        net.setInput(blob);
	
		        // Run forward pass to get raw output
		        Mat rawOutput = net.forward();

		        // Reshape the output to [8400, 85]
		        Mat reshapedOutput = rawOutput.reshape(1, new int[]{8400, 85});

		        // Decode the reshaped outputs
		        decodeYOLOXOutput(reshapedOutput, image);
	
		        // Save the output image
		        String sOutputImageFile = sOutputFileFolder+"/"+fileImage.getName()+"_output.png";

		        Imgcodecs.imwrite(sOutputImageFile, image);
			}
		}
    }

    private static void decodeYOLOXOutput(Mat output, Mat image) {
        int[] strides = {8, 16, 32};
        float confThreshold = 0.3f;  // Confidence threshold
        float nmsThreshold = 0.4f;   // Non-Maximum Suppression Threshold

        List<Rect2d> boxes = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<Integer> classIds = new ArrayList<>();

        // Iterate over each detection
        for (int i = 0; i < output.rows(); i++) {
            float[] data = new float[output.cols()];
            output.get(i, 0, data);

            // Extract the bounding box, objectness score, and class probabilities
            float cx = data[0];
            float cy = data[1];
            float width = data[2];
            float height = data[3];
            float objectness = data[4];

            // Get the class probabilities
            float[] scoresArray = new float[data.length - 5];
            System.arraycopy(data, 5, scoresArray, 0, data.length - 5);
            Mat scores = new Mat(1, scoresArray.length, CvType.CV_32F);
            scores.put(0, 0, scoresArray);

            // Use minMaxLoc to find the class with the highest score
            Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
            float maxClassScore = (float) mm.maxVal;
            int classId = (int) mm.maxLoc.x;

            // Calculate the confidence score as the product of objectness and max class score
            float confidenceScore = objectness * maxClassScore;

            if (confidenceScore > confThreshold) {  // Apply confidence threshold
                // Decode the bounding box
                int stride = strides[i / (output.rows() / strides.length)];
                int gridX = i % (output.rows() / strides.length);
                int gridY = (i / (output.rows() / strides.length)) % (output.rows() / strides.length);

                // Adjust bounding box center with respect to grid position
                cx = (cx + gridX) * stride;
                cy = (cy + gridY) * stride;
                width *= stride;
                height *= stride;

                // Correct bounding box coordinates to match original image dimensions
                cx *= (float) image.width() / 640;
                cy *= (float) image.height() / 640;
                width *= (float) image.width() / 640;
                height *= (float) image.height() / 640;

                // Convert (cx, cy, width, height) to (x1, y1, x2, y2)
                double x1 = cx - width / 2;
                double y1 = cy - height / 2;
                Rect2d box = new Rect2d(x1, y1, width, height);

                // Store the detection for NMS
                boxes.add(box);
                confidences.add(confidenceScore);
                classIds.add(classId);
            }
        }

        // Convert lists to MatOfRect2d and MatOfFloat for NMS
        MatOfRect2d matOfBoxes = new MatOfRect2d();
        matOfBoxes.fromList(boxes);
        MatOfFloat matOfConfidences = new MatOfFloat();
        matOfConfidences.fromList(confidences);
        MatOfInt indices = new MatOfInt();

        // Apply NMS using the correct types
        Dnn.NMSBoxes(matOfBoxes, matOfConfidences, confThreshold, nmsThreshold, indices);

        // Check if NMS found any indices
        if (indices.total() > 0) {
            int[] ind = indices.toArray();
            for (int i : ind) {
                Rect2d box = boxes.get(i);
                Imgproc.rectangle(image, box.tl(), box.br(), new Scalar(0, 255, 0));
                Imgproc.putText(image, "Class " + classIds.get(i) + " (" + String.format("%.2f", confidences.get(i)) + ")", 
                                box.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255));
            }
        } else {
            System.out.println("No bounding boxes selected by NMS.");
        }
    }

}
