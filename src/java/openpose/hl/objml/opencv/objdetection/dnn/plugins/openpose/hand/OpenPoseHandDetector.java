package hl.objml.opencv.objdetection.dnn.plugins.openpose.hand;


import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import hl.objml.opencv.objdetection.dnn.plugins.openpose.BaseOpenPoseDetector;
import hl.objml2.common.DetectedObj;

public class OpenPoseHandDetector extends BaseOpenPoseDetector {
	
	/**
	 *  https://github.com/CMU-Perceptual-Computing-Lab/openpose
	 *  http://vcl.snu.ac.kr/OpenPose/models/hand/pose_iter_116000.caffemodel
	 */ 
	
    @Override
	protected void decodePredictions(
	        final Mat matResult, 
	        final Mat aMatInput,
	        List<DetectedObj> aDetectedObj,
	        final double aConfidenceThreshold) {

		//1*22*46*46*CV_32FC1
		
    	int iKP = super.OBJ_CLASSESS.size();

	    Mat reshapedMat = matResult.reshape(1, new int[]{matResult.size(1), matResult.size(2), matResult.size(3)});
		
		int iH = reshapedMat.size(1);
		int iW = reshapedMat.size(2);
		
		double scaleX = (double) aMatInput.width() / iW;
	    double scaleY = (double) aMatInput.height() / iH;
	    
		for (int i = 0; i < iKP; i++) {
			// Extract heatmap for keypoint 'i'
		    Mat heatMap = reshapedMat.row(i);
		    heatMap = heatMap.reshape(1, iH);  // Reshape to 2D (46x46)
		    
	        // Find the location of the maximum confidence
	        Core.MinMaxLocResult mmr = Core.minMaxLoc(heatMap);
	        double confidence = mmr.maxVal;
	        
	        if (confidence > aConfidenceThreshold) {
	        	String label = OBJ_CLASSESS.get(i);
	            int x = (int) (mmr.maxLoc.x * scaleX);
	            int y = (int) (mmr.maxLoc.y * scaleY);

	            DetectedObj obj = new DetectedObj(i, label, new Point(x,y), confidence);
                aDetectedObj.add(obj);
                
	            // Add the detected keypoint with confidence score
	            //System.out.println(label+" "+confidence+" -> ("+x+","+y+")");
	        }
		}
		
    	ANNOTATE_OUTPUT_IMG = true;
	}
}