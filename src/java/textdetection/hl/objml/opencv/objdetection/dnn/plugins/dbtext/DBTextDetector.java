package hl.objml.opencv.objdetection.dnn.plugins.dbtext;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.dnn.TextDetectionModel;
import org.opencv.dnn.TextDetectionModel_DB;
import org.opencv.imgproc.Imgproc;

import hl.objml2.common.FrameDetectionMeta;
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
	    }
        
        // Set the input parameters
        Size inputSize = getImageInputSize();
        Scalar mean = new Scalar(122.67891434, 116.66876762, 104.00698793);
        textDetector.setInputParams(1.0 / 255.0, inputSize, mean, true);
        
		return null;
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
	        textDetector.setPreferableBackend(getDnnBackend());
	        textDetector.setPreferableTarget(getDnnTarget());
	        List<MatOfPoint> detections = new ArrayList<MatOfPoint>();
	        textDetector.detect(matOutput, detections);
        	
	        if(detections.size()>0)
	        {
	        	 if(ANNOTATE_OUTPUT_IMG)
		         {
			        // Draw detections on the image
			        for (MatOfPoint contour : detections) {
			            Imgproc.polylines(matOutput, List.of(contour), true, new Scalar(0, 255, 0), 2);
			        }
		        }
	        }

			//
	        if(matOutput!=null)
	        {
	        	frameOutput.setAnnotatedFrameImage(matOutput.clone());
				//
				FrameDetectionMeta meta = new FrameDetectionMeta();
				meta.setObjml_model_name(getModelFileName());
				meta.setObjml_plugin_name(getPluginName());
				
				frameOutput.setFrameDetectionMeta(meta);
				//
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