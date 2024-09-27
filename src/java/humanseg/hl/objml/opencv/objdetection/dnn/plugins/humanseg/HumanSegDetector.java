package hl.objml.opencv.objdetection.dnn.plugins.humanseg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.ObjDetectionBasePlugin;
import hl.opencv.util.OpenCvUtil;

public class HumanSegDetector extends ObjDetectionBasePlugin {
	
	private static Net NET_DNN 						= null;
	private static List<String> OBJ_CLASSESS 		= new ArrayList<String>();
    private static float DEF_CONFIDENCE_THRESHOLD 	= 0.5f;
    private static float DEF_NMS_THRESHOLD 			= 0.4f;
    private static Size DEF_INPUT_SIZE 				= new Size(192, 192);
    private static boolean SWAP_RB_CHANNEL			= true;
    private static boolean APPLY_IMG_PADDING 		= false;
    private static boolean ANNOTATE_OUTPUT_IMG 		= true;


	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}
	

	protected void init()
	{
		NET_DNN = Dnn.readNet( getModelFileName());
		
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
			if(sNMSThreshold!=null && sNMSThreshold.trim().length()>0)
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
		
		/////////
		// Convert to float and normalize (example values)
		//aMatInput.convertTo(aMatInput, CvType.CV_32F, 1.0 / 255, 0); // Normalize to [0,1]

		// Subtract mean and divide by standard deviation (example values)
		//Core.subtract(aMatInput, new Scalar(0.485, 0.456, 0.406), aMatInput);
		//Core.divide(aMatInput, new Scalar(0.229, 0.224, 0.225), aMatInput);
		////////
		
		
		return Dnn.blobFromImage(aMatInput, 1.0 / 255.0, sizeInput, Scalar.all(0), true, false);		
	}

	/**
	 *  https://github.com/JunkyByte/easy_ViTPose
	 *  https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx
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
		
		if(aInferenceOutputMat==null || aInferenceOutputMat.size()==0)
			return mapResult;
		
		List<Mat> outputs = aInferenceOutputMat;
		
		Size sizeDnnInput = DEF_INPUT_SIZE;
		
		// Process output
        Mat matResult = outputs.get(0);
		matResult = postProcess(matResult, sizeDnnInput);

		 // Decode detection
        float fConfidenceThreshold 	= DEF_CONFIDENCE_THRESHOLD;
        float fNMSThreshold 		= DEF_NMS_THRESHOLD;
        
        List<DetectedObj> outputKeypoints 	= new ArrayList<>();
        //
        decodePredictions(matResult, 
        		aMatInput.size(),
        		outputKeypoints, 
        		fConfidenceThreshold);
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
				mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_ANNOTATED_MAT, matOutputImg);
	        }
	        
	        mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_DETECTION_JSON, frameObjs.toJson());
			mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_TOTAL_COUNT, outputKeypoints.size());

			//
			mapResult.put(ObjDetectionBasePlugin._KEY_THRESHOLD_DETECTION, fConfidenceThreshold);
			mapResult.put(ObjDetectionBasePlugin._KEY_THRESHOLD_NMS, fNMSThreshold);
			//
        }
		return mapResult;
	}
	
	private static Mat postProcess(Mat matOutputDetections, Size sizeInput)
	{
        return matOutputDetections;
    }

	private void decodePredictions(
	        final Mat matResult, 
	        final Size aMatSize,
	        List<DetectedObj> aDetectedObj,
	        final float aConfidenceThreshold) {
	    
		System.out.println("matResult="+matResult);
		int width 	= matResult.size(2);
		int height 	= matResult.size(3);
		
		// Reshape the Mat to have shape 2x192x192
		Mat reshapedMat = matResult.reshape(1, 2 * width); 
		
		// Extract the second channel (foreground probabilities)
		Mat foreground = reshapedMat.rowRange(width, 2 * height);
		
		Mat segOutput = foreground.reshape(1, height); 
		
		Mat binaryMask = new Mat();
		Imgproc.threshold(segOutput, binaryMask, aConfidenceThreshold, 1, Imgproc.THRESH_BINARY); // Threshold at 0.5
		binaryMask.convertTo(binaryMask, CvType.CV_8UC1, 255);
		
		OpenCvUtil.resize(binaryMask, (int)aMatSize.width, (int)aMatSize.height, false);
		
		System.out.println("width="+width);
		System.out.println("height="+width);
		System.out.println("binaryMask="+segOutput);
		System.out.println();
		
		
		List<MatOfPoint> listContours = new ArrayList<>();
		Imgproc.findContours(binaryMask, listContours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
		
		if(listContours.size()>0)
		{
			for(MatOfPoint mp: listContours)
			{
				DetectedObj obj = new DetectedObj(0, "person", mp, 1.0);
				aDetectedObj.add(obj);
			}
		}
		
		
		
	}
}