package hl.objml.opencv.objdetection.dnn.plugins.ultraface;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;
import hl.objml2.plugin.ObjDetectionBasePlugin;


public class UltraFaceDetector extends ObjDetectionBasePlugin {
	
	private static Net NET_ULTRAFACE 				= null;
	private static List<String> OBJ_CLASSESS 		= new ArrayList<String>();
    private static float DEF_CONFIDENCE_THRESHOLD 	= 0.7f;
    private static float DEF_NMS_THRESHOLD 			= 0.5f;
    private static Size DEF_INPUT_SIZE 				= new Size(320,240);
    
    private static boolean SWAP_RB_CHANNEL			= false;
    private static boolean APPLY_IMG_PADDING 		= true;
    private static boolean ANNOTATE_OUTPUT_IMG 		= true;


	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}

	/**
	 *  WIP
	 *  
	 *  https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
	 *  
	 *  ONNX Model = https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/models/onnx
	 *  
	 */
	public Map<String, Object> detect(final Mat aMatInput, JSONObject aCustomThresholdJson) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		try {
			if(NET_ULTRAFACE==null)
	        {
				init();
	        }
			
			// Prepare input
			Size sizeInput 	= DEF_INPUT_SIZE;
			Mat matInputImg = aMatInput.clone();					
			Mat matDnnImg 	= preProcess(matInputImg, sizeInput, APPLY_IMG_PADDING, SWAP_RB_CHANNEL);
			NET_ULTRAFACE.setInput(matDnnImg);

//System.out.println("matDnnImg="+matDnnImg);
			
	        // Run the forward pass
	        List<Mat> outputs = new ArrayList<>();
	        List<String> outNames = new ArrayList<>();
	        outNames.add("boxes");   // Name of the bounding boxes output
	        outNames.add("scores");  // Name of the scores output
	        NET_ULTRAFACE.forward(outputs, outNames);


			 // Decode detection
	        double scaleOrgW = aMatInput.width() / sizeInput.width;
	        double scaleOrgH = aMatInput.height() / sizeInput.height;
	        float fConfidenceThreshold 	= DEF_CONFIDENCE_THRESHOLD;
	        float fNMSThreshold 		= DEF_NMS_THRESHOLD;
	        
	        List<Rect2d> outputBoxes 		= new ArrayList<>();
	        List<Float> outputConfidences 	= new ArrayList<>();
	        List<Integer> outputClassIds 	= new ArrayList<>();
	        //
	        decodePredictions(outputs, 
	        		scaleOrgW, scaleOrgH,  
	        		outputBoxes, outputConfidences, outputClassIds, 
	        		fConfidenceThreshold);
	        
//System.out.println("decodePredictions.outputBoxes.size()="+outputBoxes.size());
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
	        
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mapResult;
	}
	
	private void init()
	{
		NET_ULTRAFACE = Dnn.readNetFromONNX( getModelFileName());
		
		if(NET_ULTRAFACE!=null)
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
			//System.out.println();
			//System.out.println("*init* DEF_CONFIDENCE_THRESHOLD="+DEF_CONFIDENCE_THRESHOLD);
			//System.out.println("*init* DEF_NMS_THRESHOLD="+DEF_NMS_THRESHOLD);
			//System.out.println("*init* DEF_INPUT_SIZE="+DEF_INPUT_SIZE);

		}
	}
	
	private static Mat preProcess(Mat aMatInput, Size sizeInput, 
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

		return Dnn.blobFromImage(aMatInput, 1.0/128, sizeInput, Scalar.all(127), true, false);		
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
	        List<Mat> matOutputs, 
	        final double aScaleW,
	        final double aScaleH,
	        List<Rect2d> boxes, List<Float> confidences, List<Integer> classIds,
	        final float aConfidenceThreshold) {
	    
		
		// https://docs.openvino.ai/2024/omz_models_model_ultra_lightweight_face_detection_rfb_320.html
		// Input Mat = 1 * 3(Color) * 240(H) * 320(W)
		// Output Mat = Box:{1, 4420, 4} , Score {1, 4420, 2}
		
        Mat matBoxes 	=  matOutputs.get(0);
        Mat matScores 	=  matOutputs.get(1);
        
        int numDetections = matScores!=null?matScores.size(1):0; 
		
System.out.println("numDetections="+numDetections);
System.out.println("matBoxes.empty()="+matBoxes.empty());
System.out.println("matScores.empty()="+matScores.empty());

		if(numDetections>0)
        {
			for (int i = 0; i < numDetections; i++) {
	            double[] scoreArray = matScores.get(0, i);
	            
	            
	            if (scoreArray == null || scoreArray.length < 2) {
	                System.err.println("Error: Score array is null or has incorrect size.");
	                continue;
	            }
	            
System.out.println("scoreArray="+scoreArray.length);	            
	            
				//Get the score for the face class (index 1)
				int classId = (int) scoreArray[0];
	            float faceScore = (float) scoreArray[1];
System.out.println("faceScore="+faceScore);
	            
				double dWidth 	= 320 * aScaleW;
				double dHeight 	= 240 * aScaleH;
	            
	            if (faceScore > aConfidenceThreshold) {
	                // Get the bounding box coordinates (in normalized format)
	                double[] boxArray = matBoxes.get(1, i);
	                

	                //
	                int x1 = (int) (boxArray[0] * dWidth);
	                int y1 = (int) (boxArray[1] * dHeight);
	                //
	                int x2 = (int) (boxArray[2] * dWidth);
	                int y2 = (int) (boxArray[3] * dHeight);
	                //
	                
	                int width	= y2 - y1;
	                int height	= x2 - x1;
				
				            
		            Rect2d r = new Rect2d(x1, y1, width, height);
		            
		            classIds.add(classId);
		            confidences.add((float)faceScore);
		            boxes.add(r);
		            
			    }

	        }
	        
        }
        
		
	}
}