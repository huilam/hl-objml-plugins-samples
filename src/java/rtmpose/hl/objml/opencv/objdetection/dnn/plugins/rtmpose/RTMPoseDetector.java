package hl.objml.opencv.objdetection.dnn.plugins.rtmpose;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.MLPluginConfigProp;
import hl.objml2.plugin.MLPluginFrameOutput;
import hl.objml2.plugin.ObjDetDnnBasePlugin;
import hl.opencv.util.OpenCvUtil;


@SuppressWarnings("unused")
public class RTMPoseDetector extends ObjDetDnnBasePlugin {
	
    private static final double TAG_GROUPING_THRESHOLD = 1.0; // tune as needed!
    
	private static boolean CROP_IMAGE 			= false;
	private static boolean SWAP_RB 				= true;
    private static boolean ANNOTATE_OUTPUT_IMG 	= true;
    
    /**
     * https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
     * 
     * https://github.com/PINTO0309/PINTO_model_zoo/blob/main/269_Higher-HRNet/download.sh
     */
    
	@Override
    public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		double dScaleW = aMatInput.width() / getImageInputSize().width;
		double dScaleH = aMatInput.height() / getImageInputSize().height;
		
		int iNewH = 0;
		int iNewW = 0;
		if(dScaleW>dScaleH)
		{
			iNewW = (int)getImageInputSize().width;
		}
		else
		{
			iNewH = (int)getImageInputSize().height;
		}
		OpenCvUtil.resize(aMatInput, iNewW, iNewH, true);
		
		Mat matDnnInput = Dnn.blobFromImage(
				aMatInput, 1.0/255.0, getImageInputSize(), Scalar.all(0), SWAP_RB, CROP_IMAGE);
		aDnnNet.setInput(matDnnInput);
		List<Mat> listOutput = new ArrayList<>();
		aDnnNet.forward(listOutput, aDnnNet.getUnconnectedOutLayersNames());
		return listOutput;
	}
	
	@Override
	public MLPluginFrameOutput parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat) {

		MLPluginFrameOutput frameOutput = new MLPluginFrameOutput();
		

		double dConfidenceThreshold = getConfidenceThreshold();
		List<DetectedObj> listDetectedObj = new ArrayList<>();
		listDetectedObj = extractKeypoints(aMatInput, aInferenceOutputMat, listDetectedObj, dConfidenceThreshold);
		
		/////////////////////////////
	    
	    FrameDetectedObj frameObjs = new FrameDetectedObj();
	    for(DetectedObj obj : listDetectedObj)
	    {
	    	frameObjs.addDetectedObj(obj);
	    }

	    // Annotate the image with the detected skeletons
	    if (ANNOTATE_OUTPUT_IMG) {
	        Mat matOutputImg = DetectedObjUtil.annotateImage(aMatInput, frameObjs, null, false);
	        frameOutput.setAnnotatedFrameImage(matOutputImg);
	    }
	    
	    frameOutput.setFrameDetectedObj(frameObjs);
	    return frameOutput;
	}
	
   protected List<DetectedObj> extractKeypoints(
    		 final Mat aInputMat 
    		,final List<Mat> aOutputMatList
    		,List<DetectedObj> aDetectedObjs
	        ,final double aConfidenceThreshold)
    {
	   //TODO
	   
		Mat aMat0 = aOutputMatList.get(0);
		Mat aMat1 = aOutputMatList.get(1);
		
	   return aDetectedObjs;
    }
	
	@Override
	public MLPluginConfigProp prePropInit(MLPluginConfigProp aProps) 
	{
		return aProps;
	}

	
}