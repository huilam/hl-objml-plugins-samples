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
		double dScaleW = aMatInput.width() / DEF_INPUT_SIZE.width;
		double dScaleH = aMatInput.height() / DEF_INPUT_SIZE.height;
		
		int iNewH = 0;
		int iNewW = 0;
		if(dScaleW>dScaleH)
		{
			iNewW = (int)DEF_INPUT_SIZE.width;
		}
		else
		{
			iNewH = (int)DEF_INPUT_SIZE.height;
		}
		OpenCvUtil.resize(aMatInput, iNewW, iNewH, true);
		
		Mat matDnnInput = Dnn.blobFromImage(
				aMatInput, 1.0/255.0, DEF_INPUT_SIZE, Scalar.all(0), SWAP_RB, CROP_IMAGE);
		aDnnNet.setInput(matDnnInput);
		List<Mat> listOutput = new ArrayList<>();
		aDnnNet.forward(listOutput, aDnnNet.getUnconnectedOutLayersNames());
		return listOutput;
	}
	
	@Override
	public Map<String, Object> parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat) {
	    Map<String, Object> mapResult = new HashMap<String, Object>();

	    Mat matHeatmap 	= aInferenceOutputMat.get(1);
	    FrameDetectedObj frameObjs = getAllKeyPoints(aMatInput, matHeatmap);
	    
	    Mat matTagmap 	= aInferenceOutputMat.get(0);
	    frameObjs = groupKeypoints(frameObjs, matTagmap);

	    // Annotate the image with the detected skeletons
	    if (ANNOTATE_OUTPUT_IMG) {
	        Mat matOutputImg = DetectedObjUtil.annotateImage(aMatInput, frameObjs, null, false);
	        mapResult.put(ObjDetDnnBasePlugin._KEY_OUTPUT_FRAME_ANNOTATED_IMG, matOutputImg);
	    }

	    mapResult.put(ObjDetDnnBasePlugin._KEY_OUTPUT_FRAME_DETECTIONS, frameObjs);
	    return mapResult;
	}
	
	private FrameDetectedObj groupKeypoints(FrameDetectedObj frameObjs, Mat matTagmap)
	{
		//TODO
		return frameObjs;
	}
	
	private FrameDetectedObj getAllKeyPoints(Mat aMatInput, Mat matHeatmap)
	{
	    FrameDetectedObj frameObjs = new FrameDetectedObj();
	    
	    double aWRatio = (aMatInput.width() / DEF_INPUT_SIZE.width);
	    double aHRatio = (aMatInput.height() / DEF_INPUT_SIZE.height);
	    int iKpCount 	= matHeatmap.size(1);  // Number of keypoints (17 for COCO dataset)
	    int iHeight 	= matHeatmap.size(2);  // Heatmap height
	    int iWidth 		= matHeatmap.size(3);  // Heatmap width
	    
	    // Reshape the heatmap to (17 x 480 x 320)
	    matHeatmap = matHeatmap.reshape(1, new int[]{iKpCount, iHeight, iWidth});

	    // Loop through each keypoint type (e.g., nose, left_eye, right_eye, etc.)
	    for (int i = 0; i < iKpCount; i++) {
	        // Extract the ith keypoint's heatmap
	        Mat heatmap = matHeatmap.row(i).reshape(1, iHeight); // Shape becomes 480x320

	        // Find all local maxima as potential keypoints for this keypoint type
	        List<Point> keypointCandidates = findLocalMaxima(heatmap, getConfidenceThreshold());

	        for (Point candidate : keypointCandidates) {
	            // Calculate the confidence score for this keypoint
	            double confScore = heatmap.get((int)candidate.y, (int)candidate.x)[0];
                String objLabel = OBJ_CLASSESS.get(i);

                // Scale the keypoint coordinates back to the original image size
                double scaleX = (DEF_INPUT_SIZE.width / iWidth) * aWRatio;
                double scaleY = (DEF_INPUT_SIZE.height / iHeight) * aHRatio;
                candidate.x *= scaleX;
                candidate.y *= scaleY;

                // Create a DetectedObj for this keypoint
                DetectedObj obj = new DetectedObj(i, objLabel, candidate, confScore);
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
	public Properties prePropInit(Properties aProps) 
	{
		return aProps;
	}

	
}