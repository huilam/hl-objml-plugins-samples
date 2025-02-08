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
        
        int iTotalDetections = o1.size(1);
        int iDataSize = 18;
        
        
        Mat output1 = o1.reshape(1, new int[]{iTotalDetections, o1.size(2)});
        Mat output2 = o2.reshape(1, new int[]{iTotalDetections, iDataSize});
        
        Map<Integer, Float> mapDetections = 
        		getTopDetections(2, output1, aConfidenceThreshold);
        
        for(Object oIdx : mapDetections.keySet())
        {
        	int idx = (int)oIdx;
        	float confidence = (float) mapDetections.get(idx);
    
            float[] bestDetection = new float[iDataSize];
            output2.get(idx, 0, bestDetection);

            // Extract bounding box values
            float cx = bestDetection[0] * scaleX;
            float cy = bestDetection[1] * scaleY;
            float width  = bestDetection[2] * scaleX;
            float height = bestDetection[3] * scaleY;

            // Convert from center-based to top-left based
            float x = cx - (width/2);
            float y = cy - (height/2);

            // Clip bounding box within image bounds
            x = Math.max(0, Math.min(x, imgW - 1));
            y = Math.max(0, Math.min(y, imgH - 1));
            width = Math.min(width, imgW - x);
            height = Math.min(height, imgH - y);

            Rect2d box = new Rect2d(x, y, width, height);
            
            System.out.println("\n>> " + idx + " confidence=" + confidence + " box=" + box);

            DetectedObj obj = new DetectedObj(idx, "Hand", box, confidence);
            aDetectedObj.add(obj);
        }

        ANNOTATE_OUTPUT_IMG = true;
    }
    
    private double normalizeConfidenceScore(double rawConfidence)
    {
    	return rawConfidence > 1.0 ? (1.0 / (1.0 + Math.exp(-rawConfidence))) : rawConfidence;
    }
    
    protected Map<Integer, Float> getTopDetections(
    		int iTopN, final Mat output1,final double aConfidenceThreshold)
    {
    	if(iTopN<1)
    		iTopN = 1;
    	
    	Mat matTmpOutput = output1.clone();
    	
    	Map<Integer, Float> mapTopNDetections = new HashMap<Integer, Float> ();
    	for(int n=0; n<iTopN; n++)
    	{
    		// Use OpenCV's minMaxLoc() to find highest confidence
            Core.MinMaxLocResult mmr = Core.minMaxLoc(matTmpOutput);
            float confidence = (float) normalizeConfidenceScore(mmr.maxVal);
            
            if(confidence>aConfidenceThreshold)
            {
                int idx = (int) mmr.maxLoc.y; // Get best detection index   
                mapTopNDetections.put(Integer.valueOf(idx), Float.valueOf(confidence));
                matTmpOutput.put(idx, 0, -1);
            }
            else
            {
            	break;
            }
    	}
    	return mapTopNDetections;
    }

}