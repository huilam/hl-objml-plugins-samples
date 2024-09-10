package hl.objml.opencv.objdetection.dnn.plugins.dbtext;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.TextDetectionModel;
import org.opencv.dnn.TextDetectionModel_DB;
import org.opencv.imgproc.Imgproc;
import hl.objml2.common.DetectedObj;
import hl.objml2.plugin.ObjDetectionBasePlugin;
import hl.opencv.util.OpenCvUtil;


public class DBTextDetector extends ObjDetectionBasePlugin {

	private TextDetectionModel textDetector = null;
	
	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}

	@Override
	public Map<String, Object> detect(Mat aMatInput, JSONObject aCustomThresholdJson) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		
		if(aMatInput!=null)
		{
			mapResult = detect(aMatInput);
		}
		
		return mapResult;
	}
	
	/**
	 *  Model = https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
	 **/
	private Map<String, Object>  detect(Mat aMatInput) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		
		if(!isPluginOK() || aMatInput==null)
			return null;
		
		Mat srcImg 	= null;
		Mat faces 	= null;
		
		
		try {
	        srcImg 		= aMatInput.clone();
	        OpenCvUtil.removeAlphaChannel(srcImg);
	       
	        File fileModel 	= new File(_model_filename);
			
	        float scoreThreshold 	= 0.9f;
	        float nmsThreshold 		= 0.3f;
	                   
	        if(textDetector==null)
	        {        
	        	textDetector = new TextDetectionModel_DB(fileModel.getAbsolutePath());
		    }
	        
	        
	        // Set the input parameters
	        int inputWidth = 736;
	        int inputHeight = 736;
	        Size inputSize = new Size(inputWidth, inputHeight);
	        Scalar mean = new Scalar(122.67891434, 116.66876762, 104.00698793);
	        textDetector.setInputParams(1.0 / 255.0, inputSize, mean, true);

	       
	        // Perform text detection
	        List<MatOfPoint> detections = new ArrayList<MatOfPoint>();
	        		
	        textDetector.detect(srcImg, detections);

        	DetectedObj objs = new DetectedObj();
	        if(detections.size()>0)
	        {
		        // Draw detections on the image
		        for (MatOfPoint contour : detections) {
		            Imgproc.polylines(srcImg, List.of(contour), true, new Scalar(0, 255, 0), 2);
		            
		           //objs.addDetectedObj(0, "text", 1.0f, new Rect2d(r.x, r.y, r.width, r.height));
		        }
	        }
	        mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_TOTAL_COUNT, detections.size());
	        
	        
	        if(srcImg!=null)
	        {
	        	mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_ANNOTATED_MAT, srcImg.clone());
	        }
	        
	    	mapResult.put(ObjDetectionBasePlugin._KEY_THRESHOLD_DETECTION, scoreThreshold);
			mapResult.put(ObjDetectionBasePlugin._KEY_THRESHOLD_NMS, nmsThreshold);
        	mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_DETECTION_JSON, objs.toJson());
	        
		}finally
		{
			
			if(faces!=null)
			{
				faces.release();
			}
			
			if(srcImg!=null)
        	{
        		srcImg.release();
        	}
		}
		return mapResult;
    }

}