package hl.objml.opencv.objdetection.dnn.plugins.yolo;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import hl.objml.opencv.objdetection.MLDetectionBasePlugin;
import hl.opencv.util.OpenCvUtil;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class TestYoloX {
	
	private static final int[] STRIDES =  {8, 16, 32};

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
    
    public static void meshgrid(Mat matXCoords, Mat matYCoords, Size aSize)
    {
    	int hsize = (int)aSize.height;
    	int wsize = (int)aSize.width;
    	
        // Create the meshgrid
    	matXCoords = new Mat(hsize, wsize, CvType.CV_32F);
    	matYCoords = new Mat(hsize, wsize, CvType.CV_32F);

        for (int i = 0; i < hsize; i++) {
            for (int j = 0; j < wsize; j++) {
            	matXCoords.put(i, j, j); // X coordinates
                matYCoords.put(i, j, i); // Y coordinates
            }
        }

        // The xCoords and yCoords now represent the meshgrid    	
    }
    
    
	private static Mat postProcess(Mat matOutputDetections, Size inputSize)
	{
		//https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py
			
		int detectionCount = 0;
		int[] hSizes = new int[STRIDES.length];
		int[] wSizes = new int[STRIDES.length];
		
		for (int i=0; i<STRIDES.length; i++) {
			hSizes[i] = Math.floorDiv((int)inputSize.height, STRIDES[i]);
			wSizes[i] = Math.floorDiv((int)inputSize.width, STRIDES[i]);
			detectionCount += hSizes[i]*wSizes[i];
		}
		
		
		int[] matOutputDetectionsShape = {8400, 85};
		matOutputDetections = matOutputDetections.reshape(1, matOutputDetectionsShape);

		
		int iOutputDetectionSize = matOutputDetections.size(0);
		
		if (detectionCount != iOutputDetectionSize) {
			System.err.println("The ML model output is not as expected ! detectionCount:"+detectionCount+" != iOutputDetectionSize:"+iOutputDetectionSize);
			return null;
		}
		///////////////////////
		

		int detectionIdx = 0;
		for (int i = 0; i < STRIDES.length; i++) {
            int hSize = hSizes[i];
            int wSize = wSizes[i];
            int stride = STRIDES[i];

            for (int y = 0; y < hSize; y++) {
                for (int x = 0; x < wSize; x++) {

                    double value0 = matOutputDetections.get(detectionIdx, 0)[0];
                    value0 = (value0 + x) * stride;
                    matOutputDetections.put(detectionIdx, 0, value0);

                    double value1 = matOutputDetections.get(detectionIdx, 1)[0];
                    value1 = (value1 + y) * stride;
                    matOutputDetections.put(detectionIdx, 1, value1);

                    double value2 = matOutputDetections.get(detectionIdx, 2)[0];
                    value2 = Math.exp(value2) * stride;
                    matOutputDetections.put(detectionIdx, 2, value2);

                    double value3 = matOutputDetections.get(detectionIdx, 3)[0];
                    value3 = Math.exp(value3) * stride;
                    matOutputDetections.put(detectionIdx, 3, value3);
                    
                    detectionIdx++;
                }
            }
        }
        
        return matOutputDetections;
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

        // Prepare the input blob for the DNN
        Size sizeInput = new Size(640, 640); // Assuming input size for YOLOX

        // Load the image
        List<File> listImages = new ArrayList<File>();
        
        listImages.add(new File("./test/images/dog_bike_car.png"));
        //listImages.add(new File("./test/images/dashcam_streetview.jpg"));
        
        int iCount = 0;
        for(File fileImage : listImages)
        {
System.out.println("\n "+(++iCount)+". Processing "+fileImage.getAbsolutePath()+" ...");
	        Mat imgOrig = MLDetectionBasePlugin.getCvMatFromFile(fileImage);
			 // Convert from BGR to RGB
	        //Imgproc.cvtColor(imgOrig, imgOrig, Imgproc.COLOR_BGR2RGB);
System.out.println("      image="+matShapeToString(imgOrig));
	
	        
	        Mat blob = Dnn.blobFromImage(imgOrig, 1 / 255.0, sizeInput, Scalar.all(0), true, false);
System.out.println("      blob="+matShapeToString(blob));
	
	
	        // Set the input to the network
	        net.setInput(blob);
	
	        // Run forward pass to get output
	        List<Mat> outputs = new ArrayList<>();
	        List<String> outNames = net.getUnconnectedOutLayersNames();
	        net.forward(outputs, outNames);
	        
	        Mat matResult = outputs.get(0);
	        
System.out.println("      matOutput)="+matShapeToString(matResult));
	
			matResult = postProcess(matResult, sizeInput);
			
System.out.println("      postProcess.matOutput)="+matShapeToString(matResult));	        
	        // Post-processing the output
	        float confThreshold = 0.5f;
	        float nmsThreshold 	= 0.4f;
	
	        List<Integer> classIds = new ArrayList<>();
	        List<Float> confidences = new ArrayList<>();
	        List<Rect2d> boxes = new ArrayList<>();
	
	        // Assuming output format: [batch, num of boxes, 85] -> where last 85 corresponds to [x, y, w, h, conf, class scores]
	        int iImgH = (int) sizeInput.height;
	        int iImgW = (int) sizeInput.width;
	        
            for (int i = 0; i < matResult.rows(); i++) {
                Mat row = matResult.row(i);
                Mat scores = row.colRange(5, matResult.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                
                if (confidence > confThreshold) {
                    int classId = (int) mm.maxLoc.x;
                    float[] data = new float[4];
                    row.colRange(0, 4).get(0, 0, data);

                    int centerX = (int) (data[0]);
                    int centerY = (int) (data[1]);
                    int width 	= (int) (data[2]);
                    int height 	= (int) (data[3]);
                    
                    int left 	= centerX - width / 2;
                    int top 	= centerY - height / 2;
                    
                    top = top<0? 0: top;
                    left = left<0? 0: left;
                    width = left+width<iImgW? width: iImgW-left-1;
            		height = top+height<iImgH? height: iImgH-top-1;

                    classIds.add(classId);
                    confidences.add(confidence);
                    boxes.add(new Rect2d(left, top, width, height));
                }
            }
	
System.out.println("      detected boxes.size()="+boxes.size());
	        if(boxes.size()>0)
	        {
        		
	            // Apply Non-Maximum Suppression (NMS)
	            MatOfFloat confidencesMat = new MatOfFloat(Converters.vector_float_to_Mat(confidences));
	            Rect2d[] boxesArray = boxes.toArray(new Rect2d[boxes.size()]);
	            MatOfRect2d boxesMat = new MatOfRect2d(boxesArray);
	            
	            MatOfInt indices = new MatOfInt();
	            Dnn.NMSBoxes(boxesMat, confidencesMat, confThreshold, nmsThreshold, indices);

		        int[] ind = indices.toArray();
		        System.out.println("      NMSBoxes ind.length="+ind.length);
		        if(ind.length>0)
		        {
		        	Mat imgOutput = null;
		        	try {
			            // Draw the boxes on the image
		        		imgOutput = imgOrig.clone();
			            for (int idx : ind) {
			                
			            	
			            	Rect2d box = boxes.get(idx);
			                int classId = classIds.get(idx);
			                float confScore = confidences.get(idx);
			                
			                System.out.println(" "+classId+" "+confScore+" = "+box.tl()+" "+box.br());
			                
			                Imgproc.rectangle(imgOutput, box.tl(), box.br(), new Scalar(0, 255, 0), 2);
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
			           
			            Imgcodecs.imwrite(sImgOutputFname, imgOutput);
			            System.out.println("      Output saved ="+sImgOutputFname);
		        	}
		        	finally
		        	{
		        		if(imgOutput!=null)
		        			imgOutput.release();
	
		        	}
		        }
	        }
	    }
    }
}
