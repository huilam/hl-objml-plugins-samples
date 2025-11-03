package hl.objml.opencv.objdetection.dnn.plugins.text.detect;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
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
import org.opencv.dnn.TextDetectionModel;
import org.opencv.dnn.TextDetectionModel_DB;
import org.opencv.dnn.TextRecognitionModel;
import org.opencv.imgproc.Imgproc;

import hl.common.FileUtil;
import hl.objml.opencv.objdetection.dnn.plugins.text.recog.DBTextRecognizer;
import hl.objml.opencv.objdetection.dnn.plugins.text.recog.dev.TestDBTextRecognizer;
import hl.objml2.api.ObjMLApi;
import hl.objml2.common.DetectedObj;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.MLPluginConfigProp;
import hl.objml2.plugin.MLPluginFrameOutput;
import hl.objml2.plugin.MLPluginMgr;
import hl.objml2.plugin.ObjDetBasePlugin;
import hl.opencv.util.OpenCvFilters;


public class DBTextDetector extends ObjDetBasePlugin {

	private static boolean ANNOTATE_OUTPUT_IMG 	= true;
	private TextDetectionModel textDetector 	= null;
	private static DBTextRecognizer textRecognizer = null;
	
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
			matOutput 	= aMatInput.clone();
			
	        OpenCvFilters.grayscale(matOutput);
	        // Perform text detection
	        MatOfRotatedRect rotatedRect = getTextDetection(matOutput);
	        
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
	        	    	sLabel = "_NONE_";
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
	
	
	public TextDetectionModel initTextDetector()
	{
		if(this.textDetector==null)
        {	
			isPluginOK();
			
			File fileModel = new File(getModelFileName());
			this.textDetector = new TextDetectionModel_DB(fileModel.getAbsolutePath());
        	
            // Set the input parameters
            Size inputSize = this.getImageInputSize();
            
            Scalar mean = new Scalar(122.67891434, 116.66876762, 104.00698793);
            this.textDetector.setInputParams(1.0 / 255.0, inputSize, mean, true);
            this.textDetector.setInputCrop(true);
            
            this.textDetector.setPreferableBackend(getDnnBackend());
            this.textDetector.setPreferableTarget(getDnnTarget());
	    }
		
		return this.textDetector;
	}
	
	public MatOfRotatedRect getTextDetection(final Mat aMatInput)
	{
		MatOfRotatedRect rotatedRect = new MatOfRotatedRect();
		if(this.textDetector==null)
		{
			initTextDetector();
		}
		textDetector.detectTextRectangles(aMatInput, rotatedRect);
		return rotatedRect;
	}
	
	private String doImageRecog(final Mat aInputMat, RotatedRect aRotatedRect, MLPluginFrameOutput aFrameDetectedObj)
	{
		String sLabel = null;
		if(textRecognizer!=null)
		{
			if(textRecognizer.isPluginOK())
			{
				sLabel = "_UNKNOWN_";
			}
		}
    	return sLabel;
	}
	
	
	public void setDBTextRecognizer(DBTextRecognizer aTextRecognizer)
	{
		textRecognizer = aTextRecognizer;
	}
	
}