package hl.objml.opencv.objdetection.dnn.plugins.mediapipe.hand;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect2d;
import hl.objml.opencv.objdetection.dnn.plugins.mediapipe.BaseMediaPipeDetector;
import hl.objml2.common.DetectedObj;

public class MediaPipeHandDetector extends BaseMediaPipeDetector {
	
	/**
	 *  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
	 */ 
	
    @Override
    protected void decodePredictions(
            final List<Mat> listMatOutputs, 
            final Mat aMatInput,
            List<DetectedObj> aDetectedObj,
            final double aConfidenceThreshold) {

        Mat o1 = listMatOutputs.get(0);  // 1×2016×1 = Confidence scores
        Mat o2 = listMatOutputs.get(1);  // 1×2016×18 = Bounding boxes & keypoints 

        float imgW = aMatInput.width();
        float imgH = aMatInput.height();
        
        float modelW = (float) DEF_INPUT_SIZE.width;
        float modelH = (float) DEF_INPUT_SIZE.height;
        
        float scaleX = imgW / modelW;
        float scaleY = imgH / modelH;
        
        Mat output1 = o1.reshape(1, new int[]{2016, 1});
        Mat output2 = o2.reshape(1, new int[]{2016, 18});
        
        boolean isGetBestDetection = true;
        
        Map<Integer,Float> mapDetections = 
        		isGetBestDetection ?
        				getBestDetection(output1, aConfidenceThreshold):
        				getAllDetections(output1, aConfidenceThreshold);
        
        for(Object oIdx : mapDetections.keySet())
        {
        	int idx = (int)oIdx;
        	float confidence = (float) mapDetections.get(idx);
    
            float[] bestDetection = new float[18];
            output2.get(idx, 0, bestDetection);

            // Extract bounding box values
            float cx = bestDetection[0];
            float cy = bestDetection[1];
            float width  = bestDetection[2];
            float height = bestDetection[3];

            // Convert from center-based to top-left based
            float x = cx - (width/2);
            float y = cy - (height/2);

            // Scale to original image dimensions
            x *= scaleX;
            y *= scaleY;
            width *= scaleX;
            height *= scaleY;

            // Clip bounding box within image bounds
            x = Math.max(0, Math.min(x, imgW - 1));
            y = Math.max(0, Math.min(y, imgH - 1));
            width = Math.min(width, imgW - x);
            height = Math.min(height, imgH - y);

            Rect2d box = new Rect2d(x, y, width, height);
            
            System.out.println();
            System.out.println(">> " + idx + " confidence=" + confidence + " box=" + box);

            DetectedObj obj = new DetectedObj(idx, "Hand", box, confidence);
            aDetectedObj.add(obj);
            
        }

        ANNOTATE_OUTPUT_IMG = true;
    }
    
    private double normalizeConfidenceScore(double rawConfidence)
    {
    	return rawConfidence > 1.0 ? (1.0 / (1.0 + Math.exp(-rawConfidence))) : rawConfidence;
    }
    
    private Map<Integer, Float> getAllDetections(
    		Mat output1,final double aConfidenceThreshold)
    {
    	Map<Integer, Float> mapDetections = new HashMap<Integer, Float>();
    	// Iterate through all detections
        for (int idx = 0; idx < 2016; idx++) {
            double[] confidenceArray = output1.get(idx, 0);
            if (confidenceArray == null || confidenceArray.length == 0) continue;

            float confidence = (float) normalizeConfidenceScore(confidenceArray[0]);

            if (confidence > aConfidenceThreshold) {
            	mapDetections.put(Integer.valueOf(idx), Float.valueOf(confidence));
            }
        }
        return mapDetections;
        
    }
    
    private Map<Integer, Float> getBestDetection(
    		Mat output1,final double aConfidenceThreshold)
    {
    	Map<Integer, Float> mapDetections = new HashMap<Integer, Float> ();
    	
    	// Use OpenCV's minMaxLoc() to find highest confidence
        Core.MinMaxLocResult mmr = Core.minMaxLoc(output1);
        float confidence = (float) normalizeConfidenceScore(mmr.maxVal);
        
        if(confidence>aConfidenceThreshold)
        {
            int idx = (int) mmr.maxLoc.y; // Get best detection index   
            mapDetections.put(Integer.valueOf(idx), Float.valueOf(confidence));
        }
        return mapDetections;
    }

}