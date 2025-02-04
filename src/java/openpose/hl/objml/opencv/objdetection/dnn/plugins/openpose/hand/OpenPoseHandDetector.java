package hl.objml.opencv.objdetection.dnn.plugins.openpose.hand;


import java.util.List;
import org.opencv.core.Mat;
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
		System.out.println("matResult="+matResult);
		
		Mat output = matResult;
		
		int H = output.size(2);
        int W = output.size(3);
        int nPoints = super.OBJ_CLASSESS.size();
		
		
		System.out.println("H="+H);
		System.out.println("W="+W);
		System.out.println("Keypoints="+nPoints);
		
		/**
        
        Pending decoding logic

		**/
		
	}
}