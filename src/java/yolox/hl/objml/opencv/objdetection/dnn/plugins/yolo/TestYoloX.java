package hl.objml.opencv.objdetection.dnn.plugins.yolo;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import hl.opencv.util.OpenCvUtil;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class TestYoloX {

    private static String matShapeToString(Mat mat) {
    	
    	StringBuffer sb = new StringBuffer();
    	sb.append("{");
    	for(int s=0; s<mat.dims(); s++)
    	{
    		if(s>0)
    			sb.append(",");
    		sb.append(mat.size(s));
    	}
    	sb.append("}");
        return sb.toString();
    }
    

    public static void main(String[] args) {
    	
        // Load OpenCV native library
    	 OpenCvUtil.initOpenCV();

    	 String sPackageNage = TestYoloX.class.getPackage().getName();
    	 String ssPackageFolder = sPackageNage.replaceAll("\\.", "\\/");
    	 
        // Load YOLOX ONNX model
        File fileModel = new File("./src/java/yolox/"+ssPackageFolder+"/yolox_s.onnx");
        
        System.out.println(fileModel.getAbsolutePath()+" = "+fileModel.isFile());
        Net net = Dnn.readNetFromONNX(fileModel.getAbsolutePath());

        // Load the image
        File fileImage = new File("./test/images/dog_bike_car.png");
        Mat image = Imgcodecs.imread(fileImage.getAbsolutePath());
System.out.println("image="+matShapeToString(image));

        // Prepare the input blob for the DNN
        Size sz = new Size(640, 640); // Assuming input size for YOLOX
        Mat blob = Dnn.blobFromImage(image, 1 / 255.0, sz, Scalar.all(0), true, false);
System.out.println("blob="+matShapeToString(blob));


        // Set the input to the network
        net.setInput(blob);

        // Run forward pass to get output
        List<Mat> outputs = new ArrayList<>();
        List<String> outNames = net.getUnconnectedOutLayersNames();
        net.forward(outputs, outNames);
System.out.println("outputs.get(0))="+matShapeToString(outputs.get(0)));
        
        // Post-processing the output
        float confThreshold = 0.1f;
        float nmsThreshold = 0.4f;

        List<Integer> classIds = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<Rect> boxes = new ArrayList<>();

        // Assuming output format: [batch, num of boxes, 85] -> where last 85 corresponds to [x, y, w, h, conf, class scores]
        for (Mat result : outputs) {
        	
            for (int i = 0; i < result.rows(); i++) {
                Mat row = result.row(i);
                Mat scores = row.colRange(5, result.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                
                if (confidence > confThreshold) {
                    int classId = (int) mm.maxLoc.x;
                    float[] data = new float[4];
                    row.colRange(0, 4).get(0, 0, data);

                    int centerX = (int) (data[0] * image.cols());
                    int centerY = (int) (data[1] * image.rows());
                    int width = (int) (data[2] * image.cols());
                    int height = (int) (data[3] * image.rows());
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.add(classId);
                    confidences.add(confidence);
                    boxes.add(new Rect(left, top, width, height));
                }
            }
        }

        if(boxes.size()>0)
        {
            // Apply Non-Maximum Suppression (NMS)
            MatOfFloat confidencesMat = new MatOfFloat(Converters.vector_float_to_Mat(confidences));
            Rect2d[] boxesArray = boxes.toArray(new Rect2d[0]);
            MatOfRect2d boxesMat = new MatOfRect2d(boxesArray);
            MatOfInt indices = new MatOfInt();
            Dnn.NMSBoxes(boxesMat, confidencesMat, confThreshold, nmsThreshold, indices);

            // Draw the boxes on the image
            int[] ind = indices.toArray();
            for (int idx : ind) {
                Rect box = boxes.get(idx);
                Imgproc.rectangle(image, box.tl(), box.br(), new Scalar(0, 255, 0), 2);
            }

            // Save the output image
            String sImgOutputFname = fileImage.getAbsolutePath();
           
            String sFName = fileImage.getName();
            int iExt = sFName.indexOf(".");
            if(iExt>-1)
            {
            	String sOutputFolder = fileImage.getParentFile().getAbsolutePath();
            	String sOutputFName = sFName.substring(0,iExt);
            	String sOutputExt = sFName.substring(iExt);
            	
            	sImgOutputFname = sOutputFolder+"/"+sOutputFName+"_output"+sOutputExt;
            }
           
            
            Imgcodecs.imwrite(sImgOutputFname, image);
            System.out.println("Output saved ="+sImgOutputFname);
        }
        
        System.out.println();
        System.out.println("detected boxes.size()="+boxes.size());

    }
}
