package hl.objml.opencv.objdetection.dnn.plugins.openpose.body;


import java.util.List;
import java.util.Map;

import org.opencv.core.Mat;
import org.opencv.core.Point;

import hl.objml.opencv.objdetection.dnn.plugins.openpose.BaseOpenPoseDetector;
import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;

public class OpenPoseBodyDetector extends BaseOpenPoseDetector {
	
	
	protected static boolean CALC_PAF = false;
	
	/**
	 *  https://github.com/CMU-Perceptual-Computing-Lab/openpose
	 *  http://vcl.snu.ac.kr/OpenPose/models/pose/pose_iter_584000.caffemodel
	 */ 
	
    @Override
	protected void decodePredictions(
	        final Mat matResult, 
	        final Mat aMatInput,
	        List<DetectedObj> aDetectedObjs,
	        final double aConfidenceThreshold) {
	 	
		//1*78*46*46*CV_32FC1
		
	    Mat reshapedMat = matResult.reshape(1, new int[]{matResult.size(1), matResult.size(2), matResult.size(3)});
		
      	int iKP = super.getObjClassesOfInterest().length; //25
      	int iPAF = reshapedMat.size(0)-iKP;
		int iH = reshapedMat.size(1);
		int iW = reshapedMat.size(2);
		
		//System.out.println("iKP="+iKP+", iPAF="+iPAF);
		//System.out.println("iH="+iH+", iW="+iW);
		
		aDetectedObjs = extractKeypoints(reshapedMat, aMatInput, aDetectedObjs, aConfidenceThreshold);
		if(CALC_PAF)
		{
			aDetectedObjs = calcPAFs(reshapedMat, aDetectedObjs, 
					super.getSupportedObjectPAFs());	
		}
		
    	ANNOTATE_OUTPUT_IMG = true;
	}
    
    
    protected List<DetectedObj> extractKeypoints(
    		final Mat aReshapedMat 
    		,final Mat aMatInput 
    		,List<DetectedObj> aDetectedObjs
	        ,final double aConfidenceThreshold)
    {
    	int iKP = super.getSupportedObjectLabels().length; //25
      	int iPAF = aReshapedMat.size(0)-iKP;
		int iH = aReshapedMat.size(1);
		int iW = aReshapedMat.size(2);
		
		System.out.println("getObjClassesOfInterest().length="+super.getSupportedObjectLabels().length);
		
		//System.out.println("iKP="+iKP+", iPAF="+iPAF);
		//System.out.println("iH="+iH+", iW="+iW);
		
    	double scaleX = (double) aMatInput.width() / iW;
	    double scaleY = (double) aMatInput.height() / iH;
	    
    	for (int i = 0; i < iKP; i++) 
    	{
			
			// Extract heatmap for keypoint 'i'
		    Mat heatMap = aReshapedMat.row(i);
		    heatMap = heatMap.reshape(1, iH);  // Reshape to 2D (46x46)
		    
		    //System.out.println("heatMap--->"+heatMap);
		    
		    Map<Point, Float> mapTopDetections = 
		    		DetectedObjUtil.getTopDetectionsFor2DMat(1, heatMap, aConfidenceThreshold);
		    
		    for(Object oIdx : mapTopDetections.keySet())
	        {
	        	Point pt = (Point)oIdx;
	        	float confidence = (float) mapTopDetections.get(pt);
	    
	            int x = (int) (pt.x * scaleX);
	            int y = (int) (pt.y * scaleY);

	            String label = getObjClassLabel(i);
	            DetectedObj obj = new DetectedObj(i, label, new Point(x,y), confidence);
	            
	            //System.out.println(obj);
	            aDetectedObjs.add(obj);
	        }
		    
		}
    	return aDetectedObjs;
    }
    
    
    protected static double normalizeScoreTo1(double rawScore)
    {
    	return rawScore > 1.0 ? (1.0 / (1.0 + Math.exp(-rawScore))) : rawScore;
    }
	
}