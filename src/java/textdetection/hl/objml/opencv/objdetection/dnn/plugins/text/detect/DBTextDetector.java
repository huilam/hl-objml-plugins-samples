package hl.objml.opencv.objdetection.dnn.plugins.text.detect;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRotatedRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.dnn.Net;
import org.opencv.dnn.TextDetectionModel;
import org.opencv.dnn.TextDetectionModel_DB;
import org.opencv.imgproc.Imgproc;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.MLPluginFrameOutput;
import hl.objml2.plugin.ObjDetBasePlugin;
import hl.opencv.util.OpenCvFilters;


public class DBTextDetector extends ObjDetBasePlugin {

	private static boolean ANNOTATE_OUTPUT_IMG 	= true;
	private static boolean isForceGrayScale		= true;
	private TextDetectionModel textDetector 	= null;
	
	/**
	 *  Model = https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
	 **/
	
	@Override
    public List<Mat> doInference(final Mat aMatInput, Net aDnnNet)
	{
        if(textDetector==null)
        {
        	textDetector = initTextDetector();
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
	        // Perform text detection
	        MatOfRotatedRect rotatedRect = getTextDetection(aMatInput);
	        
	        if(!rotatedRect.empty())
	        {
	        	//Create detected obj
		        FrameDetectedObj detectedObjs = new FrameDetectedObj();
		        
		        // Draw detections on the image
	        	List<MatOfPoint> contours = new ArrayList<>();
	        	for (RotatedRect rRect : rotatedRect.toList()) {
		        	Point[] pts = new Point[4];
		        	rRect.points(pts); // get 4 corner points
	        	    
		        	/**
		        	Rect rect = rRect.boundingRect();
		        	pts = rectToPoints(rect);
		        	**/
	        	    
	        	    MatOfPoint contour = new MatOfPoint(pts);
	        	    contours.add(contour);
		        	detectedObjs.addDetectedObj(new DetectedObj(0, "", contour, 1.0d));
	        	}
	        	
	        	if(ANNOTATE_OUTPUT_IMG)
		        {
	        		 matOutput = aMatInput.clone();
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
	
	private static Point[] rectToPoints(Rect rect) {
		Point[] pts = new Point[4];
		pts[0]=new Point(rect.x, rect.y); // Top-left
		pts[1]=new Point(rect.x + rect.width, rect.y); // Top-right
		pts[2]=new Point(rect.x + rect.width, rect.y + rect.height); // Bottom-right
		pts[3]=new Point(rect.x, rect.y + rect.height); // Bottom-left
        return pts;
    }
	
	public TextDetectionModel initTextDetector()
	{
		if(this.textDetector==null)
        {	
			isPluginOK();
			
			File fileModel = new File(getModelFileName());
			this.textDetector = new TextDetectionModel_DB(fileModel.getAbsolutePath());
        	
            // Set the input parameters
            //this.textDetector.setInputParams(1.0 / 255.0, inputSize, mean, true);
            this.textDetector.setInputSize(this.getImageInputSize());
            this.textDetector.setInputScale(new Scalar(1.0 / 255.0));
            this.textDetector.setInputMean(new Scalar(122.67891434, 116.66876762, 104.00698793));
            this.textDetector.setInputCrop(false);
            
            this.textDetector.setPreferableBackend(getDnnBackend());
            this.textDetector.setPreferableTarget(getDnnTarget());
	    }
		
		return this.textDetector;
	}
	
	public MatOfRotatedRect getTextDetection(final Mat aMatInput)
	{
		Mat matDetect = null;
		try {
			matDetect = aMatInput.clone();
			
			if(isForceGrayScale)
				OpenCvFilters.grayscale(matDetect);
			
			MatOfRotatedRect rotatedRect = new MatOfRotatedRect();
			if(this.textDetector==null)
			{
				initTextDetector();
			}
			textDetector.detectTextRectangles(matDetect, rotatedRect);
			
			return rotatedRect;
		}finally
		{
			if(matDetect!=null)
				matDetect.release();
		}
	}
}