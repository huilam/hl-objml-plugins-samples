package hl.objml.opencv.objdetection.dnn.plugins.openpose;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.MLPluginFrameOutput;
import hl.objml2.plugin.ObjDetDnnBasePlugin;

public class BaseOpenPoseDetector extends ObjDetDnnBasePlugin {
	
	protected static boolean SWAP_RB_CHANNEL		= false;
    protected static boolean APPLY_IMG_PADDING 		= false;
    protected static boolean ANNOTATE_OUTPUT_IMG 	= true;

	/**
	 *  https://github.com/CMU-Perceptual-Computing-Lab/openpose
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
			Size sizeDnnInput = getImageInputSize();
			
			matDnnImg = doInferencePreProcess(matInputImg, sizeDnnInput, APPLY_IMG_PADDING, SWAP_RB_CHANNEL);
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
	public MLPluginFrameOutput parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		MLPluginFrameOutput frameOutput = new MLPluginFrameOutput();
		List<Mat> outputs = aInferenceOutputMat;
		
		// Process output
        Mat matResult = outputs.get(0);
        double dConfidenceThreshold = getConfidenceThreshold();
        List<DetectedObj> outputKeypoints 	= new ArrayList<>();
        //
        decodePredictions(matResult, 
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
				frameOutput.setAnnotatedFrameImage(matOutputImg);
	        }
			frameOutput.setFrameDetectedObj(frameObjs);
			//
        }
		return frameOutput;
	}
	
	private static Mat doInferencePreProcess(Mat aMatInput, Size sizeInput, 
			boolean isApplyImgPadding, boolean isSwapRBChannel)
	{
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

		return Dnn.blobFromImage(aMatInput, 1.0 / 255.0, sizeInput, Scalar.all(0), isSwapRBChannel, false);		
	}
	
	
	protected List<DetectedObj> calcPAFs(
			final Mat aMatReshaped
			,List<DetectedObj> aDetectedObjs
			,List<int[]> aPAFList)
	{
		Map<Integer, List<DetectedObj>> mapDetectedObjs = new HashMap<Integer, List<DetectedObj>>();
		
		for( DetectedObj obj : aDetectedObjs)
		{
			Integer iKPId = obj.getObj_classid();
			List<DetectedObj> listObj = mapDetectedObjs.get(iKPId);
			if(listObj==null)
				listObj = new ArrayList<DetectedObj>();
			listObj.add(obj);
			mapDetectedObjs.put(iKPId, listObj);
		}
		return calcPAFs(aMatReshaped, mapDetectedObjs, aPAFList);
		
	}
	
	protected List<DetectedObj> calcPAFs(
			final Mat aMatReshaped
			,Map<Integer, List<DetectedObj>> aMapDetectedObjs
			,List<int[]> aPAFList)
	{
		List<DetectedObj> aDetectedObjs = new ArrayList<DetectedObj>();
		
		int iKP = super.OBJ_CLASSESS.size(); //25
      	int iPAF = aMatReshaped.size(0)-iKP;
		
      	
		// Part/Pair Affinity Fields (PAFs)
		for(int[] iPairKP : aPAFList)
		{
			int iP1 = iPairKP[0];
			int iP2 = iPairKP[1];
			
			// TODO
		}
		
		return aDetectedObjs;
	}
	

	protected void decodePredictions(
	        final Mat matResult, 
	        final Mat aMatInput,
	        List<DetectedObj> aDetectedObjs,
	        final double aConfidenceThreshold) {
	}
}