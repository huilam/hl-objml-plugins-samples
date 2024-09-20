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
import hl.objml2.plugin.ObjDetectionBasePlugin;
import hl.opencv.util.OpenCvUtil;


public class DBTextDetector extends ObjDetectionBasePlugin {

	private TextDetectionModel textDetector = null;
	
	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}
	

	/**
	 *  Model = https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
	 **/
	
	@Override
	public List<Mat> doInference(Mat aMatInput, JSONObject aCustomThresholdJson)
	{
		File fileModel 	= new File(_model_filename);
                   
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
        
		return null;
	}
	
	@Override
	public Map<String,Object> parseDetections(
			List<Mat> aInferenceOutputMat, 
			Mat aMatInput, JSONObject aCustomThresholdJson)
	{
		Map<String, Object> mapResult = new HashMap<String, Object>();
		
		if(!isPluginOK() || aMatInput==null)
			return null;
		
		Mat matOutput 	= null;
		
		try {
			matOutput 	= aMatInput.clone();
	        OpenCvUtil.removeAlphaChannel(matOutput);
	        
	        // Perform text detection
	        List<MatOfPoint> detections = new ArrayList<MatOfPoint>();
	        textDetector.detect(matOutput, detections);
        	
	        if(detections.size()>0)
	        {
		        // Draw detections on the image
		        for (MatOfPoint contour : detections) {
		            Imgproc.polylines(matOutput, List.of(contour), true, new Scalar(0, 255, 0), 2);
		        }
	        }
	        
	        mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_TOTAL_COUNT, detections.size());

	        if(matOutput!=null)
	        {
	        	mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_ANNOTATED_MAT, matOutput.clone());
	        }
	        
	        //
			mapResult.put(ObjDetectionBasePlugin._KEY_THRESHOLD_DETECTION, -1);
			mapResult.put(ObjDetectionBasePlugin._KEY_THRESHOLD_NMS, -1);
	        
		}finally
		{
			
			if(matOutput!=null)
        	{
				matOutput.release();
        	}
		}
		return mapResult;
	}
	

}