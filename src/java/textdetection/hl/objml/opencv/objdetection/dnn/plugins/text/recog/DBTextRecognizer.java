package hl.objml.opencv.objdetection.dnn.plugins.text.recog;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRotatedRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.dnn.TextRecognitionModel;
import org.opencv.imgproc.Imgproc;

import hl.objml.opencv.objdetection.dnn.plugins.text.detect.DBTextDetector;
import hl.objml2.common.DetectedObj;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.MLPluginFrameOutput;
import hl.objml2.plugin.ObjDetBasePlugin;
import hl.opencv.util.OpenCvFilters;


public class DBTextRecognizer extends ObjDetBasePlugin {

	private static boolean ANNOTATE_OUTPUT_IMG 	= true;
	private TextRecognitionModel textRecog 		= null;
	
	private DBTextDetector textDetectorPlugin 	= new DBTextDetector();
	private boolean isSkipTextDetection 		= false;
	
	/**
	 *  Model = https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
	 **/
	
	@Override
    public List<Mat> doInference(final Mat aMatInput, Net aDnnNet)
	{
		File fileModel 	= new File(_model_filename);
        
        if(textRecog==null)
        {
        	textRecog = new TextRecognitionModel(fileModel.getAbsolutePath());
        	textRecog.setVocabulary(Arrays.asList(getSupportedObjLabels()));
        	textRecog.setDecodeType("CTC-greedy");
        	// crnn_cs.onnx 	(isSwapRB=false)
        	// crnn_cs_CN.onnx  (isSwapRB=true);
        	boolean isSwapRB = true;//(sRecogModelFPath.indexOf("CN.onnx")>-1);
        	System.out.println("isSwapRB="+isSwapRB);
        	textRecog.setInputParams((
        			1.0 / 127.5), 
        			new Size(100, 32), 
        			new Scalar(127.5, 127.5, 127.5),
        			isSwapRB);
        }
        
        if(!isSkipTextDetection)
        {
        	if(textDetectorPlugin==null)
        	{
        		textDetectorPlugin 	= new DBTextDetector();
        		textDetectorPlugin.initTextDetector();
        	}
        }
        
		return new ArrayList<Mat>();
	}
	
	@Override
    public MLPluginFrameOutput parseDetections(final Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		MLPluginFrameOutput frameOutput = new MLPluginFrameOutput();
		
		if(!isPluginOK() || aMatInput==null)
			return null;
		
		Mat matOutput 	= null;
		
		try {
			matOutput 	= aMatInput.clone();
			
	        OpenCvFilters.grayscale(matOutput);
	        
	        MatOfRotatedRect rotatedRect = doTextDetection(aMatInput);
	        
	        if(!rotatedRect.empty())
	        {
	        	//Create detected obj
		        FrameDetectedObj detectedObjs = new FrameDetectedObj();
		        
		        // Draw detections on the image
	        	List<MatOfPoint> contours = new ArrayList<>();
	        	for (RotatedRect rect : rotatedRect.toList()) {
	        		
	        		
	        	    String sLabel = doImageRecog(matOutput, rect, frameOutput);
	        	    //
	        	    if(sLabel==null)
	        	    	sLabel = "_UNKNOWN_";
		        	Point[] pts = new Point[4];
	        	    rect.points(pts); // get 4 corner points

	        	    MatOfPoint contour = new MatOfPoint(pts);
	        	    contours.add(contour);
		        	detectedObjs.addDetectedObj(new DetectedObj(0, sLabel, contour, 1.0d));
	        	}
	        	
	        	if(ANNOTATE_OUTPUT_IMG)
		        {
	        		 Imgproc.polylines(matOutput, contours, true, new Scalar(0, 255, 0), 2);
		        }
	        	frameOutput.setFrameDetectedObj(detectedObjs);
	        }

	        if(matOutput!=null)
	        {
	        	frameOutput.setAnnotatedFrameImage(matOutput.clone());
	        }
	        
		}finally
		{
			
			if(matOutput!=null)
        	{
				matOutput.release();
        	}
		}
		return frameOutput;
	}
	
	private MatOfRotatedRect doTextDetection(final Mat aInputMat)
	{
		MatOfRotatedRect matRotatedRect = null;
		if(isSkipTextDetection || textDetectorPlugin==null)
		{
			// Define center point, size, and angle
	        double width = aInputMat.width();
	        double height = aInputMat.height();
	        
	        // Create the RotatedRect
	        RotatedRect rotatedRect = new RotatedRect(
	        	    new Point(width / 2.0, height / 2.0), //Center
	        	    new Size(width, height),// Size
	        	    0.0 // Angle
	        	);
	        
	        List<RotatedRect> listRotatedRect = new ArrayList<>();
	        listRotatedRect.add(rotatedRect);
	        
	        matRotatedRect = new MatOfRotatedRect();
	        matRotatedRect.fromList(listRotatedRect);
		}
		else
		{
			matRotatedRect = textDetectorPlugin.getTextDetection(aInputMat);
		}
		
		return matRotatedRect;
	}
	
	private String doImageRecog(final Mat aInputMat, RotatedRect aRotatedRect, MLPluginFrameOutput aFrameDetectedObj)
	{
		String sLabel = null;
		Mat matROI = null;
    	try{
    		matROI = getRotatedROI(aInputMat, aRotatedRect);
        	if(!matROI.empty())
        	{
        		sLabel = textRecog.recognize(matROI);
        		
        		Rect rect = aRotatedRect.boundingRect();
        		
//System.out.println("sLabel-->"+sLabel);
        		aFrameDetectedObj.putFrameOutputCustomObj("cropped_"
        				+Math.max(0, rect.x)+"x"+Math.max(0, rect.y)
        				+"_"+sLabel, matROI.clone());
        		
        	}
    	}finally
    	{
        	if(matROI!=null)
        		matROI.release();
    	}
    	return sLabel;
	}
	
	private static Mat getRotatedROI(Mat src, RotatedRect rect) {
		Mat rotationMatrix = null;
		Mat rotated = new Mat();
		try {
		    // Get the rotation matrix for the rect
		    rotationMatrix = Imgproc.getRotationMatrix2D(rect.center, rect.angle, 1.0);
	
		    // Compute the size of the rotated image
		    Size size = src.size();
		    rotated = new Mat();
		    Imgproc.warpAffine(src, rotated, rotationMatrix, size, Imgproc.INTER_CUBIC);
	
		    // Now crop the upright region corresponding to the original rotated rect
		    Size rectSize 	= rect.size;
		    rectSize.width 	= rectSize.width + 10; //extra 10px
		    rectSize.height = rectSize.height + 10; //extra 10px
		    Rect roi = new Rect(
		        (int)(rect.center.x - rectSize.width / 2),
		        (int)(rect.center.y - rectSize.height / 2),
		        (int)rectSize.width,
		        (int)rectSize.height
		    );
	
		    // Ensure ROI is within image bounds
		    roi = adjustRectToFit(roi, rotated.size());
		    return new Mat(rotated, roi);
		}
		finally
		{
			if(rotationMatrix!=null)
				rotationMatrix.release();
			
			if(rotated!=null)
				rotated.release();
		}
	}

	private static Rect adjustRectToFit(Rect rect, Size size) {
	    int x = Math.max(rect.x, 0);
	    int y = Math.max(rect.y, 0);
	    int width = Math.min(rect.width, (int)size.width - x);
	    int height = Math.min(rect.height, (int)size.height - y);
	    return new Rect(x, y, width, height);
	}
}