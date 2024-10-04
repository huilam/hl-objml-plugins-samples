package hl.objml.opencv.objdetection.dnn.plugins.higherhrnet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.ObjDetDnnBasePlugin;
import hl.opencv.util.OpenCvUtil;


public class HigherHRNetPoseDetector extends ObjDetDnnBasePlugin {
	
    private static boolean ANNOTATE_OUTPUT_IMG 		= true;
	
    
    /**
     * https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
     * 
     * https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/269_Higher-HRNet/resources.tar.gz
     */
    
	@Override
    public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
	
		OpenCvUtil.resize(aMatInput,
				(int)DEF_INPUT_SIZE.width, (int)DEF_INPUT_SIZE.height, true);
		
		
		Mat matDnnInput = Dnn.blobFromImage(
				aMatInput, 1.0/255.0, DEF_INPUT_SIZE, Scalar.all(0), true, false);
		aDnnNet.setInput(matDnnInput);
		List<Mat> listOutput = new ArrayList<>();
		aDnnNet.forward(listOutput, aDnnNet.getUnconnectedOutLayersNames());
		return listOutput;
	}
	
	@Override
    public Map<String,Object> parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		Map<String, Object> mapResult = new HashMap<String, Object>();
		
		
		Mat matHigherResHeatmap = aInferenceOutputMat.get(1);
		int iKpCount 	= matHigherResHeatmap.size(1);
		int iHeight 	= matHigherResHeatmap.size(2);
		int iWidth 		= matHigherResHeatmap.size(3);

		System.out.println(" iKpCount 	= "+iKpCount);
		System.out.println(" iWidth 	= "+iWidth);
		System.out.println(" iHeight 	= "+iHeight);

	    // No need to reshape twice, reshape into (17 x 480 x 320)
	    matHigherResHeatmap = matHigherResHeatmap.reshape(1, new int[]{iKpCount, iHeight, iWidth});

		System.out.println(" reshaped-matHigherResHeatmap 	= "+matHigherResHeatmap);
		
		double aWRatio = (aMatInput.width() / DEF_INPUT_SIZE.width);
		double aHRatio = (aMatInput.height() / DEF_INPUT_SIZE.height);
		
		FrameDetectedObj frameObjs = new FrameDetectedObj();
		for (int i = 0; i < iKpCount; i++) 
		{	
	        // Extract the ith keypoint's heatmap
	        Mat heatmap = matHigherResHeatmap.row(i).reshape(1, iHeight); // Shape becomes 480x320

	        // Find the maximum value and its location in the heatmap
	        Core.MinMaxLocResult mmr = Core.minMaxLoc(heatmap);
	        double confScore = mmr.maxVal;  // Confidence score
	        
	        if(confScore>getConfidenceThreshold())
	        {
		        Point keypoint = mmr.maxLoc;  // Refined keypoint location
		        String objLabel = OBJ_CLASSESS.get(i);
	    
			    double scaleX 	= (DEF_INPUT_SIZE.width / iWidth) * aWRatio;
			    double scaleY 	= (DEF_INPUT_SIZE.height / iHeight) * aHRatio;
			    keypoint.x *= scaleX;
			    keypoint.y *= scaleY;
			    
			    DetectedObj obj = new DetectedObj(i, objLabel, keypoint, confScore);
			    frameObjs.addDetectedObj(obj);
	        }
		}
			        
        // Draw bounding boxes
		if(ANNOTATE_OUTPUT_IMG)
        {
			Mat matOutputImg = DetectedObjUtil.annotateImage(aMatInput, frameObjs, null, false);
			mapResult.put(ObjDetDnnBasePlugin._KEY_OUTPUT_FRAME_ANNOTATED_IMG, matOutputImg);
        }
        mapResult.put(ObjDetDnnBasePlugin._KEY_OUTPUT_FRAME_DETECTIONS, frameObjs);
        
        return mapResult;
	}
	
	@Override
	public Properties prePropInit(Properties aProps) 
	{
		return aProps;
	}

	
}