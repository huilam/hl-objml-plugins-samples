package hl.objml.opencv.objdetection.dnn.plugins.higherhrnet;

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
public class HigherHRNetPoseDetector extends ObjDetDnnBasePlugin {
	
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
		
		System.out.println();
		System.out.println(aInferenceOutputMat.get(0));
		System.out.println(aInferenceOutputMat.get(1));		
		
		Map<String, Object> mapResult = new HashMap<String, Object>();

		Mat matTagmap 	= aInferenceOutputMat.get(0);
	    Mat matHeatmap 	= aInferenceOutputMat.get(1);
	    FrameDetectedObj frameObjs = getAllKeyPoints(aMatInput, matHeatmap, matTagmap);
	    
	    frameObjs = groupKeypoints(aMatInput, matTagmap, frameObjs);

	    // Annotate the image with the detected skeletons
	    if (ANNOTATE_OUTPUT_IMG) {
	        Mat matOutputImg = DetectedObjUtil.annotateImage(aMatInput, frameObjs, null, false);
	        frameOutput.setAnnotatedFrameImage(matOutputImg);
	    }
	    
	    frameOutput.setFrameDetectedObj(frameObjs);
	    return frameOutput;
	}
	
	private FrameDetectedObj groupKeypoints(Mat aMatInput, Mat matTagmap, FrameDetectedObj frameObjs)
	{
	    int iKpCount 	= matTagmap.size(1);  // Number of keypoints 
	    int iHeight 	= matTagmap.size(2);  // Heatmap height
	    int iWidth 		= matTagmap.size(3);  // Heatmap width
		
	    // Reshape the matTagmap to (34 x 80 x 120)
	    matTagmap = matTagmap.reshape(1, new int[]{iKpCount, iHeight, iWidth});
	    
	    Map<Double, DetectedObj> mapDetectedObj = new TreeMap<>();
	    
	    for(String sObjClassName : frameObjs.getObjClassNames())
	    {
		    List<DetectedObj> listObj = frameObjs.getDetectedObjByObjClassName(sObjClassName);
		    System.out.print(" "+sObjClassName+" ("+listObj.size()+")");
	    	for(DetectedObj obj : listObj)
	    	{
	    		double dTagValue = Double.parseDouble(obj.getObj_trackingid());
	    		
	    		System.out.printf("   - %.8f\n",dTagValue);
                mapDetectedObj.put(dTagValue, obj);
	    	}
	    }
    	
	    
		//TODO
		return frameObjs;
	}
	
	private FrameDetectedObj getAllKeyPoints(Mat aMatInput, Mat matHeatmap, Mat matTagmap)
	{
	    FrameDetectedObj frameObjs = new FrameDetectedObj();
	    
	    int iKpCount 	= matHeatmap.size(1);  // Number of keypoints (17 for COCO dataset)
	    int iHeight 	= matHeatmap.size(2);  // Heatmap height
	    int iWidth 		= matHeatmap.size(3);  // Heatmap width
	    
	    double aWRatio = (aMatInput.width() / getImageInputSize().width);
	    double aHRatio = (aMatInput.height() / getImageInputSize().height);
        double scaleX = (getImageInputSize().width / iWidth) * aWRatio;
        double scaleY = (getImageInputSize().height / iHeight) * aHRatio;	    
        
	    // Reshape the heatmap to (17 x 480 x 320)
	    matHeatmap = matHeatmap.reshape(1, new int[]{iKpCount, iHeight, iWidth});
	    
	    // Loop through each keypoint type (e.g., nose, left_eye, right_eye, etc.)
	    for (int i = 0; i < iKpCount; i++) {
	        // Extract the ith keypoint's heatmap
	        Mat heatmap = matHeatmap.row(i).reshape(1, iHeight); // Shape becomes 480x320

	        // Find all local maxima as potential keypoints for this keypoint type
	        List<Point> keypointCandidates = findLocalMaxima(heatmap, getConfidenceThreshold());

	        for (Point pt : keypointCandidates) {
	        	int iX = (int)pt.x;
	        	int iY = (int)pt.y;
	        	
	            // Calculate the confidence score for this keypoint
	            double confScore = heatmap.get(iY, iX)[0];
                String objLabel = OBJ_CLASSESS.get(i);
                
                //get TagValue
                int iKpIdx = i * 2; //tagMap 34 = keypoint 17
                int itagY = iY / 2;
                int itagX = iX / 2;
                double tagValX = matTagmap.get(new int[]{0, iKpIdx, itagY, itagX})[0];
	            double tagValY = matTagmap.get(new int[]{0, iKpIdx+1, itagY, itagX})[0];
	            //double dKpTagValue = Math.sqrt(Math.pow(tagValX, 2) + Math.pow(tagValY, 2));
	            double dKpTagValue = tagValX;// + (tagValY);
                
	            System.out.printf("tagValX= %8f\n",tagValX);
	            System.out.printf("tagValY= %8f\n",tagValY);
	            System.out.printf("dKpTagValue= %8f\n\n",dKpTagValue);
	            
                // Scale the keypoint coordinates back to the original image size
	            pt.x *= scaleX;
	            pt.y *= scaleY;

                // Create a DetectedObj for this keypoint
                DetectedObj obj = new DetectedObj(i, objLabel, pt, confScore);
                obj.setObj_trackingid(String.valueOf(dKpTagValue));
                frameObjs.addDetectedObj(obj);
	        }
	    }
		return frameObjs;
	}

	/**
	 * Find local maxima in the heatmap.
	 */
	private List<Point> findLocalMaxima(Mat heatmap, double threshold) {
	    List<Point> maxima = new ArrayList<>();

	    // Traverse the heatmap and find local maxima
	    for (int y = 1; y < heatmap.rows() - 1; y++) {
	        for (int x = 1; x < heatmap.cols() - 1; x++) {
	            double centerValue = heatmap.get(y, x)[0];

	            // Check if this point is a local maximum and above the threshold
	            if (centerValue > threshold &&
	                centerValue > heatmap.get(y - 1, x)[0] && centerValue > heatmap.get(y + 1, x)[0] &&
	                centerValue > heatmap.get(y, x - 1)[0] && centerValue > heatmap.get(y, x + 1)[0]) {
	                
	                maxima.add(new Point(x, y));
	            }
	        }
	    }
	    return maxima;
	}

	
	@Override
	public MLPluginConfigProp prePropInit(MLPluginConfigProp aProps) 
	{
		return aProps;
	}

	
}