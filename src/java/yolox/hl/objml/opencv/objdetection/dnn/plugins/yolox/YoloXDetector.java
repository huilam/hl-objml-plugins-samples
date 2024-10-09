package hl.objml.opencv.objdetection.dnn.plugins.yolox;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Core;
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
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.ObjDetDnnBasePlugin;


public class YoloXDetector extends ObjDetDnnBasePlugin {
	
    private static boolean SWAP_RB_CHANNEL			= false;
    private static boolean APPLY_IMG_PADDING 		= false;
    private static boolean ANNOTATE_OUTPUT_IMG 		= true;

	/**
	 *  ONNX Model = https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime
	 *  
	 *  Processing Reference
	 *  - https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py
	 *  - https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py
	 *  - https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py
	 */
	@Override
    public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		List<Mat> outputs = null;
		Mat matInputImg = null;
		Mat matDnnImg = null;
		try {
			
			// Prepare input
			matInputImg = aMatInput.clone();					
			matDnnImg = preProcess(matInputImg, DEF_INPUT_SIZE, APPLY_IMG_PADDING, SWAP_RB_CHANNEL);
			aDnnNet.setInput(matDnnImg);

	        // Run inference
	        outputs = new ArrayList<>();
	        aDnnNet.forward(outputs, aDnnNet.getUnconnectedOutLayersNames());
		}
		finally
		{
			if(matInputImg!=null)
				matInputImg.release();
			
			if(matDnnImg!=null)
				matDnnImg.release();
		}
		return outputs;
			
	}
	
	@Override
    public Map<String,Object> parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		Map<String, Object> mapResult = new HashMap<String, Object>();
		
		
		Mat matResult = aInferenceOutputMat.get(0);
		matResult = postProcess(matResult, DEF_INPUT_SIZE);
		
		 // Decode detection
        double scaleOrgW = aMatInput.width() / DEF_INPUT_SIZE.width;
        double scaleOrgH = aMatInput.height() / DEF_INPUT_SIZE.height;
        double fConfidenceThreshold 	= DEF_CONFIDENCE_THRESHOLD;
        double fNMSThreshold 			= DEF_NMS_THRESHOLD;
        
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
	        FrameDetectedObj frameObjs = new FrameDetectedObj();
	        for (int idx : indices) {
	        	
	            Rect2d box 			= outputBoxes.get(idx);
	            int classId 		= outputClassIds.get(idx);
	            String classLabel 	= OBJ_CLASSESS.get(classId);
	            Float confScore 	= outputConfidences.get(idx);
	            
	            DetectedObj obj = new DetectedObj(classId, classLabel, box, confScore);
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
		
		
		return Dnn.blobFromImage(aMatInput, 1, sizeInput, Scalar.all(0), true, false);		
	}
	
	private static Mat postProcess(Mat matOutputDetections, Size sizeInput)
	{
		int detectionCount = 0;
		matOutputDetections = matOutputDetections.reshape(1, new int[] {8400, 85});

		int[] STRIDES 	=  {8, 16, 32};
		int[] hSizes 	= new int[STRIDES.length];
		int[] wSizes 	= new int[STRIDES.length];
		
		for (int i=0; i<STRIDES.length; i++) {
			hSizes[i] = Math.floorDiv((int)sizeInput.height, STRIDES[i]);
			wSizes[i] = Math.floorDiv((int)sizeInput.width, STRIDES[i]);
			//
			detectionCount += hSizes[i]*wSizes[i];
		}

		if (detectionCount != matOutputDetections.size(0)) {
			return null;
		}
		///////////////////////

		int detectionIdx = 0;
		for (int i = 0; i < STRIDES.length; i++) {
            int hSize = hSizes[i];
            int wSize = wSizes[i];
            int stride = STRIDES[i];

            for (int y = 0; y < hSize; y++) {
                for (int x = 0; x < wSize; x++) {

                    double value0 = (matOutputDetections.get(detectionIdx, 0)[0] + x) * stride;
                    matOutputDetections.put(detectionIdx, 0, value0);

                    double value1 = (matOutputDetections.get(detectionIdx, 1)[0] + y) * stride;
                    matOutputDetections.put(detectionIdx, 1, value1);

                    double value2 = Math.exp(matOutputDetections.get(detectionIdx, 2)[0]) * stride;
                    matOutputDetections.put(detectionIdx, 2, value2);

                    double value3 = Math.exp(matOutputDetections.get(detectionIdx, 3)[0]) * stride;
                    matOutputDetections.put(detectionIdx, 3, value3);
                    
                    detectionIdx++;
                }
            }
        }
        
        return matOutputDetections;
    }

	
	private static int[] applyNMS(List<Rect2d> aBoxesList, List<Float> aConfidencesList, 
			double CONFIDENCE_THRESHOLD, double NMS_THRESHOLD)
	{
        MatOfInt indices = new MatOfInt();

        if(aBoxesList.size()>0)
        {
	        // Apply Non-Maximum Suppression
	        MatOfRect2d boxesMat = new MatOfRect2d();
	        boxesMat.fromList(aBoxesList);
	        
	        MatOfFloat confidencesMat = new MatOfFloat();
	        confidencesMat.fromList(aConfidencesList);
	        
	        Dnn.NMSBoxes(boxesMat, confidencesMat, (float)CONFIDENCE_THRESHOLD, (float)NMS_THRESHOLD, indices);
        }
        return indices.toArray();

	}

	private boolean isObjOfInterest(int aObjClassId)
	{
		String sObjClassName = OBJ_CLASSESS.get(aObjClassId);
		return isObjOfInterest(sObjClassName);
	}
	
	private boolean isObjOfInterest(String aObjClassName)
	{
		return super.isObjClassOfInterest(aObjClassName);
	}

	private void decodePredictions(
	        final Mat matResult, 
	        final double aScaleW,
	        final double aScaleH,
	        List<Rect2d> boxes, List<Float> confidences, List<Integer> classIds,
	        final double aConfidenceThreshold) {
	    
        for (int i = 0; i < matResult.rows(); i++) 
        {
            Mat row = matResult.row(i);
            Mat scores = row.colRange(5, matResult.cols());
            Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
            float confidence = (float) mm.maxVal;
            
            if (confidence >= aConfidenceThreshold) {
                int classId = (int) mm.maxLoc.x;

                //check if obj of interest
        		if(isObjOfInterest(classId))
        		{
                    float[] data = new float[4];
                    row.colRange(0, 4).get(0, 0, data);

                    double centerX = data[0] * aScaleW;
                    double centerY = data[1] * aScaleH;
                    double width   = data[2] * aScaleW;
                    double height  = data[3] * aScaleH;
                    
                    double left = centerX - (width / 2);
                    double top 	= centerY - (height / 2);
                    
                    long lLeft  	= (long) Math.floor((left<0? 0: left));
                    long lTop  		= (long) Math.floor((top<0? 0: top));
                    long lWidth 	= (long) Math.floor(width);
            		long lHeight 	= (long) Math.floor(height);
            		
	                classIds.add(classId);
	                confidences.add(confidence);
	                boxes.add(new Rect2d(lLeft, lTop, lWidth, lHeight));
        		}
            }
        }
	}
}