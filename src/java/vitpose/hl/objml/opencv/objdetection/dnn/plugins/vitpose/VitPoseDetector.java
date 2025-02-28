package hl.objml.opencv.objdetection.dnn.plugins.vitpose;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.ObjDetDnnBasePlugin;

public class VitPoseDetector extends ObjDetDnnBasePlugin {
	
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
			if(aDnnNet==null)
	        {
				init();
	        }
			
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
		List<Mat> outputs = aInferenceOutputMat;
		
		// Process output
        Mat matResult = outputs.get(0);
        double dConfidenceThreshold = getConfidenceThreshold();
        List<DetectedObj> outputKeypoints 	= new ArrayList<>();
        //
        decodePredictions(matResult, 
        		aMatInput.size(),
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
	    
		Mat output = matResult;
		
		// Extract the heatmaps (output has shape [1, 17 or 25, 64, 48])
        int numKeypoints = (int) output.size(1); // Number of keypoints (17 / 25)
        int height = output.size(2); // Height of heatmaps (64)
        int width = output.size(3); // Width of heatmaps (48)

        double dScaleW = aMatSize.width / width;
        double dScaleH = aMatSize.height / height;
        
        
        for (int i = 0; i < numKeypoints; i++) {
            // Extract the i-th channel (heatmap)
            Mat heatmap = new Mat();
            output.row(0).col(i).reshape(1, height).copyTo(heatmap);

            // Find the peak point in the heatmap
            Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(heatmap);

            double confidence = minMaxLocResult.maxVal;
            
            if (confidence >= aConfidenceThreshold) 
            {
                Point peak = minMaxLocResult.maxLoc;
                double x = peak.x * dScaleW;
                double y = peak.y * dScaleH;
                String label = OBJ_CLASSESS.get(i);
                
            	DetectedObj obj = new DetectedObj(i, label, new Point(x,y), confidence);
                aDetectedObj.add(obj);
            }
            
        }
		
	}
}