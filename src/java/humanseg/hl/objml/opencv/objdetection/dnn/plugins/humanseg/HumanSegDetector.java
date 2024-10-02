package hl.objml.opencv.objdetection.dnn.plugins.humanseg;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
import hl.objml2.plugin.ObjDetDnnBasePlugin;
import hl.opencv.util.OpenCvUtil;

public class HumanSegDetector extends ObjDetDnnBasePlugin {
	
	private static boolean SWAP_RB_CHANNEL			= true;
    private static boolean APPLY_IMG_PADDING 		= false;
    private static boolean ANNOTATE_OUTPUT_IMG 		= true;

	/**
	 *  https://github.com/JunkyByte/easy_ViTPose
	 *  https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx
	 */
	@Override
	public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		List<Mat> outputs 	= null;
		Mat matInputImg 	= null;
		Mat matDnnImg 		= null;
		try {
			
			// Prepare input
			matInputImg = aMatInput.clone();					
			Size sizeDnnInput = DEF_INPUT_SIZE;
			
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
	public Map<String,Object> parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		Map<String, Object> mapResult = new HashMap<String, Object>();
		
		if(aInferenceOutputMat==null || aInferenceOutputMat.size()==0)
			return mapResult;
		
		List<Mat> outputs = aInferenceOutputMat;
		
		
		// Process output
        Mat matResult = outputs.get(0);
        double fConfidenceThreshold 	= super.DEF_CONFIDENCE_THRESHOLD;
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
		
		return Dnn.blobFromImage(aMatInput, 1.0 / 255.0, sizeInput, Scalar.all(0), true, false);		
	}

	private void decodePredictions(
	        final Mat matResult, 
	        final Size aMatSize,
	        List<DetectedObj> aDetectedObj,
	        final double aConfidenceThreshold) {
	    
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