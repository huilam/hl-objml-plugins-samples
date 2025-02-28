package hl.objml.opencv.objdetection.dnn.plugins.mediapipe;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.ObjDetDnnBasePlugin;

public class BaseMediaPipeDetector extends ObjDetDnnBasePlugin {
	
	protected static boolean SWAP_RB_CHANNEL		= true;
    protected static boolean ANNOTATE_OUTPUT_IMG 	= true;

	/**
	 *  https://github.com/google-ai-edge/mediapipe
	 */ 
	
    @Override
    public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		List<Mat> outputs 	= null;
		Mat matInputImg 	= null;
		Mat matDnnImg 		= null;
		try {
			if(aDnnNet==null)
	        {
				init();
	        }
			
			// Prepare input
			matInputImg = aMatInput.clone();
			Size sizeDnnInput = DEF_INPUT_SIZE;
			
			matDnnImg = doInferencePreProcess(matInputImg, sizeDnnInput, true, SWAP_RB_CHANNEL);
			aDnnNet.setInput(matDnnImg);

	        // Run inference
			outputs = new ArrayList<>();
			aDnnNet.forward(outputs, aDnnNet.getUnconnectedOutLayersNames());

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		finally
		{
			if(matDnnImg!=null)
				matDnnImg.release();
		}
			
		return outputs;
	}
	
	@Override
	public Map<String,Object> parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		Map<String, Object> mapResult = new HashMap<String, Object>();
		List<Mat> listMatOutputs = aInferenceOutputMat;
		
		// Process output
        double dConfidenceThreshold = getConfidenceThreshold();
        List<DetectedObj> outputKeypoints 	= new ArrayList<>();
        //
        decodePredictions(listMatOutputs, 
        		aMatInput,
        		outputKeypoints, 
        		dConfidenceThreshold);
        //
        if(outputKeypoints.size()>0)
        {
	        // Calculate bounding boxes
        	FrameDetectedObj frameObjs = new FrameDetectedObj();
	        for (DetectedObj obj : outputKeypoints) {
	            frameObjs.addDetectedObj(obj);
	        }
	        
	        // Draw bounding boxes
			if(ANNOTATE_OUTPUT_IMG)
	        {
				Mat matOutputImg = DetectedObjUtil.annotateImage(aMatInput, frameObjs, null, false);
				mapResult.put(ObjDetDnnBasePlugin._KEY_OUTPUT_FRAME_ANNOTATED_IMG, matOutputImg);
	        }
	        mapResult.put(ObjDetDnnBasePlugin._KEY_OUTPUT_FRAME_DETECTIONS, frameObjs);
			//
        }
		return mapResult;
	}
	
	private static Mat doInferencePreProcess(Mat aMatInput, Size sizeInput, 
			boolean isApplyImgPadding, boolean isSwapRBChannel)
	{
		//System.out.println();
		//System.out.println("sizeInput="+sizeInput);
		//System.out.println("before aMatInput="+aMatInput.size());
		
		if(isApplyImgPadding)
		{
			Mat matPaddedImg = null;
			Mat matROI = null;
			try {
				
				int iMaxPixels = Math.max(aMatInput.width(), aMatInput.height());
				matPaddedImg = new Mat(new Size(iMaxPixels,iMaxPixels), aMatInput.type(), Scalar.all(0));
				matROI = matPaddedImg.submat(0,aMatInput.rows(),0,aMatInput.cols());
				aMatInput.copyTo(matROI);
				
				aMatInput = matPaddedImg.clone();
			}
			finally
			{
				if(matPaddedImg!=null)
					matPaddedImg.release();
				if(matROI!=null)
					matROI.release();
			}
		}
		//System.out.println("after aMatInput="+aMatInput.size());

		return Dnn.blobFromImage(aMatInput, 1.0/255.0, sizeInput, Scalar.all(0), isSwapRBChannel, false);		
	}

	protected void decodePredictions(
	        final List<Mat> listMatOutputs, 
	        final Mat aMatInput,
	        List<DetectedObj> aDetectedObj,
	        final double aConfidenceThreshold) {
	}
    
    protected Map<Point, Float> getTopDetections(
    		int iTopN, final Mat matOutput,final double aConfidenceThreshold)
    {
    	
    	return DetectedObjUtil.getTopDetectionsFor2DMat(iTopN, matOutput, aConfidenceThreshold);
    }

}