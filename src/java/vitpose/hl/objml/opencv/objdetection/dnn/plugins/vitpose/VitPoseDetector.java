package hl.objml.opencv.objdetection.dnn.plugins.vitpose;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;
import hl.objml2.plugin.ObjDetectionBasePlugin;


public class VitPoseDetector extends ObjDetectionBasePlugin {
	
	private static Net NET_DNN 						= null;
	private static List<String> OBJ_CLASSESS 		= new ArrayList<String>();
    private static float DEF_CONFIDENCE_THRESHOLD 	= 0.9f;
    private static float DEF_NMS_THRESHOLD 			= 0.8f;
    private static Size DEF_INPUT_SIZE 				= new Size(192,256);
    
    private static boolean SWAP_RB_CHANNEL			= true;
    private static boolean APPLY_IMG_PADDING 		= true;
    private static boolean ANNOTATE_OUTPUT_IMG 		= true;


	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}

	/**
	 *  ONNX Model = https://github.com/Pukei-Pukei/ViTPose-ONNX/tree/main
	 *             = https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx/coco
	 *  
	 *  Processing Reference
	 *  - https://github.com/Pukei-Pukei/ViTPose-ONNX/blob/main/utils/vitpose_util.py
	 */
	
	@Override
	public List<Mat> doInference(Mat aMatInput, JSONObject aCustomThresholdJson)
	{
		List<Mat> outputs 	= null;
		Mat matInputImg 	= null;
		Mat matDnnImg 		= null;
		try {
			if(NET_DNN==null)
	        {
				init();
	        }
			
			// Prepare input
			matInputImg = aMatInput.clone();					
			Size sizeDnnInput = DEF_INPUT_SIZE;
			
			matDnnImg = doInferencePreProcess(matInputImg, sizeDnnInput, APPLY_IMG_PADDING, SWAP_RB_CHANNEL);
			NET_DNN.setInput(matDnnImg);

	        // Run inference
			outputs = new ArrayList<>();
			NET_DNN.forward(outputs, NET_DNN.getUnconnectedOutLayersNames());

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
	public Map<String,Object> parseDetections(
			List<Mat> aInferenceOutputMat, 
			Mat aMatInput, JSONObject aCustomThresholdJson)
	{
		Map<String, Object> mapResult = new HashMap<String, Object>();
		List<Mat> outputs = aInferenceOutputMat;
		
		Size sizeDnnInput = DEF_INPUT_SIZE;
		
		// Process output
        Mat matResult = outputs.get(0);
		matResult = postProcess(matResult, sizeDnnInput);

		 // Decode detection
        double scaleOrgW = aMatInput.width() / sizeDnnInput.width;
        double scaleOrgH = aMatInput.height() / sizeDnnInput.height;
        float fConfidenceThreshold 	= DEF_CONFIDENCE_THRESHOLD;
        float fNMSThreshold 		= DEF_NMS_THRESHOLD;
        
        List<Rect2d> outputBoxes 		= new ArrayList<>();
        List<Float> outputConfidences 	= new ArrayList<>();
        List<Integer> outputClassIds 	= new ArrayList<>();
        //
        decodePredictions(matResult, 
        		scaleOrgW, scaleOrgH,  
        		outputBoxes, outputConfidences, outputClassIds, 
        		fConfidenceThreshold);
        //
        if(outputBoxes.size()>0)
        {
	        // Apply NMS
	        int[] indices = applyNMS(outputBoxes, outputConfidences, fConfidenceThreshold, fNMSThreshold);

	        // Calculate bounding boxes
	        DetectedObj objs = new DetectedObj();
	        for (int idx : indices) {
	        	
	            Rect2d box 			= outputBoxes.get(idx);
	            int classId 		= outputClassIds.get(idx);
	            String classLabel 	= OBJ_CLASSESS.get(classId);
	            Float confScore 	= outputConfidences.get(idx);
	            
	            objs.addDetectedObj(classId, classLabel, confScore, box);
	        }
	        
	        // Draw bounding boxes
			if(ANNOTATE_OUTPUT_IMG)
	        {
				Mat matOutputImg = DetectedObjUtil.annotateImage(aMatInput, objs);
				mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_ANNOTATED_MAT, matOutputImg);
	        }
	        
	        mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_DETECTION_JSON, objs.toJson());
			mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_TOTAL_COUNT, indices.length);

			//
			mapResult.put(ObjDetectionBasePlugin._KEY_THRESHOLD_DETECTION, fConfidenceThreshold);
			mapResult.put(ObjDetectionBasePlugin._KEY_THRESHOLD_NMS, fNMSThreshold);
			//
        }
		return mapResult;
	}
	
	protected void init()
	{
		NET_DNN = Dnn.readNetFromONNX( getModelFileName());
		
		if(NET_DNN!=null)
		{
			String sSupporedLabels = (String) getPluginProps().get("objml.mlmodel.detection.support-labels");
			if(sSupporedLabels!=null)
			{
				String[] objs = sSupporedLabels.split("\n");
				OBJ_CLASSESS = new ArrayList<>(Arrays.asList(objs));
			}
			//
			String sConfThreshold = (String) getPluginProps().get("objml.mlmodel.detection.confidence-threshold");
			if(sConfThreshold!=null)
			{
				try {
					DEF_CONFIDENCE_THRESHOLD = Float.parseFloat(sConfThreshold);
				}catch(NumberFormatException ex)
				{
					ex.printStackTrace();
				}
			}
			//
			String sNMSThreshold = (String) getPluginProps().get("objml.mlmodel.detection.nms-threshold");
			if(sNMSThreshold!=null)
			{
				try {
					DEF_NMS_THRESHOLD = Float.parseFloat(sNMSThreshold);
				}catch(NumberFormatException ex)
				{
					ex.printStackTrace();
				}
			}
			//
			String sInputImageSize = (String) getPluginProps().get("objml.mlmodel.detection.input-size");
			if(sInputImageSize!=null)
			{

				String sSeparator = "x";
				if(sInputImageSize.indexOf(sSeparator)==-1)
					sSeparator = ",";
				
				double dWidth = 0;
				double dHeight = 0;
				String[] sSize = sInputImageSize.split(sSeparator);
				if(sSize.length>0)
				{
					try {
						dWidth 	= Double.parseDouble(sSize[0]);
						dHeight = dWidth;
						if(sSize.length>1)
						{
							dHeight = Double.parseDouble(sSize[1]);
						}
					}
					catch(NumberFormatException ex)
					{
						ex.printStackTrace();
					}
					DEF_INPUT_SIZE = new Size(dWidth,dHeight);
				}
						
			}
		}
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
		
		// Convert from BGR to RGB
		if(isSwapRBChannel)
		{
			Imgproc.cvtColor(aMatInput, aMatInput, Imgproc.COLOR_BGR2RGB);
		}
		
		
		return Dnn.blobFromImage(aMatInput, 1, sizeInput, Scalar.all(0), true, false);		
	}
	
	private static Mat postProcess(Mat matOutputDetections, Size sizeInput)
	{
        return matOutputDetections;
    }
	

	
	private static int[] applyNMS(List<Rect2d> aBoxesList, List<Float> aConfidencesList, float CONFIDENCE_THRESHOLD, float NMS_THRESHOLD)
	{
        MatOfInt indices = new MatOfInt();

        if(aBoxesList.size()>0)
        {
	        // Apply Non-Maximum Suppression
	        MatOfRect2d boxesMat = new MatOfRect2d();
	        boxesMat.fromList(aBoxesList);
	        
	        MatOfFloat confidencesMat = new MatOfFloat();
	        confidencesMat.fromList(aConfidencesList);
	        
	        Dnn.NMSBoxes(boxesMat, confidencesMat, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
        }
        return indices.toArray();

	}

	private void decodePredictions(
	        final Mat matResult, 
	        final double aScaleW,
	        final double aScaleH,
	        List<Rect2d> boxes, List<Float> confidences, List<Integer> classIds,
	        final float aConfidenceThreshold) {
	    
		Mat output = matResult;
		
		// Extract the heatmaps (output has shape [1, 17, 64, 48])
        int numKeypoints = (int) output.size(1); // Number of keypoints (17)
        int height = output.size(2); // Height of heatmaps (64)
        //int width = output.size(3); // Width of heatmaps (48)

        for (int i = 0; i < numKeypoints; i++) {
            // Extract the i-th channel (heatmap)
            Mat heatmap = new Mat();
            output.row(0).col(i).reshape(1, height).copyTo(heatmap);

            // Find the peak point in the heatmap
            Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(heatmap);
            Point peak = minMaxLocResult.maxLoc;

            double x = peak.x;
            double y = peak.y;
            double confidence = minMaxLocResult.maxVal;
            
            System.out.printf("Keypoint %d: x=%.2f, y=%.2f, confidence=%.2f%n", i, x, y, confidence);
            
            if (confidence >= aConfidenceThreshold) {
            	
            	
            }

            
        }
		
	}
}