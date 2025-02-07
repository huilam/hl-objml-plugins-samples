package hl.objml.opencv.objdetection.dnn.plugins.mediapipe.hand;

import java.util.List;
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

		
    	
        Mat o1 = listMatOutputs.get(0);  // 1*2016* 1*CV_32FC1 = Confidence scores
        Mat o2 = listMatOutputs.get(1);  // 1*2016*18*CV_32FC1 = Bounding boxes & keypoints 
        
	    Mat output1 = o1.reshape(1, new int[] {1, 2016});
	    Mat output2 = o2.reshape(1, new int[] {2016,18});

        // Use OpenCV's minMaxLoc() to find highest confidence
        Core.MinMaxLocResult mmr = Core.minMaxLoc(output1);
        double maxConfidence = mmr.maxVal;
        int bestIndex = (int) mmr.maxLoc.y; // Get best detection index
        
        if (maxConfidence > aConfidenceThreshold && bestIndex < 2016) {
        	float[] bestDetection = new float[18];
            output2.get(bestIndex, 0, bestDetection);

            // Ensure values are in range [0,1] before scaling
            float x = Math.max(0, Math.min(bestDetection[0], 1)) * aMatInput.cols();
            float y = Math.max(0, Math.min(bestDetection[1], 1)) * aMatInput.rows();
            float width = Math.max(0, Math.min(bestDetection[2], 1)) * aMatInput.cols();
            float height = Math.max(0, Math.min(bestDetection[3], 1)) * aMatInput.rows();

            // Ensure box stays within the image
            x = Math.max(0, x);
            y = Math.max(0, y);
            width = Math.min(aMatInput.cols() - x, width);
            height = Math.min(aMatInput.rows() - y, height);
            
            String label = OBJ_CLASSESS.get(bestIndex);
            Rect2d box = new Rect2d(x,y,width,height);
            
            DetectedObj obj = new DetectedObj(bestIndex, label, box, maxConfidence);
            aDetectedObj.add(obj);
            
        }
		
    	ANNOTATE_OUTPUT_IMG = true;
	}
}