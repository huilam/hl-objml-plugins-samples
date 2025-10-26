package hl.objml.opencv.objdetection.dnn.plugins.text.detect;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRotatedRect;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.dnn.TextDetectionModel;
import org.opencv.dnn.TextDetectionModel_DB;
import org.opencv.imgproc.Imgproc;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.MLPluginFrameOutput;
import hl.objml2.plugin.ObjDetBasePlugin;
import hl.opencv.util.OpenCvUtil;


public class DBTextDetector extends ObjDetBasePlugin {

	private static boolean ANNOTATE_OUTPUT_IMG 	= true;
	private TextDetectionModel textDetector 	= null;
	

	/**
	 *  Model = https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
	 **/
	
	@Override
    public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		File fileModel 	= new File(_model_filename);
                   
        if(textDetector==null)
        {        
        	textDetector = new TextDetectionModel_DB(fileModel.getAbsolutePath());
        	
            // Set the input parameters
            Size inputSize = getImageInputSize();
            
            Scalar mean = new Scalar(122.67891434, 116.66876762, 104.00698793);
            textDetector.setInputParams(1.0 / 255.0, inputSize, mean, true);
            textDetector.setInputCrop(true);
            
            textDetector.setPreferableBackend(getDnnBackend());
	        textDetector.setPreferableTarget(getDnnTarget());
	    }
        
        
		return new ArrayList<Mat>();
	}
	
	@Override
    public MLPluginFrameOutput parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		MLPluginFrameOutput frameOutput = new MLPluginFrameOutput();
		
		if(!isPluginOK() || aMatInput==null)
			return null;
		
		Mat matOutput 	= null;
		
		try {
			matOutput 	= aMatInput.clone();
	        OpenCvUtil.removeAlphaChannel(matOutput);
	        
	        // Perform text detection
	        MatOfRotatedRect rects = new MatOfRotatedRect();
	        textDetector.detectTextRectangles(matOutput, rects);
	        
	        List<MatOfPoint> polygons = new ArrayList<>();
	        
	        List<RotatedRect> detections = rects.toList();
	        if(detections.size()>0)
	        {
	        	FrameDetectedObj detectedObjs = new FrameDetectedObj();
	        	
		        // Draw detections on the image
		        for (RotatedRect rotatedRect : detections) 
		        {
		        	//if(rotatedRect.size.area()<20)
		        	//	continue;
		        	
		        	Point[] ptRect = new Point[4];
		        	rotatedRect.points(ptRect);
		        	
		        	MatOfPoint pts = new MatOfPoint(ptRect);
		            polygons.add(pts);
		            
		            detectedObjs.addDetectedObj(new DetectedObj(0, "text", pts, 1.0d));
		        }
		        if(ANNOTATE_OUTPUT_IMG && polygons.size()>0)
	            {
	            	Imgproc.polylines(matOutput, polygons, true, new Scalar(0, 255, 0), 2);
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
	
}