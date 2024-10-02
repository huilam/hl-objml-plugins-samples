package hl.objml.opencv.objdetection.dnn.plugins.yunet.face;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.ObjDetBasePlugin;
import hl.objml2.plugin.ObjDetDnnBasePlugin;
import hl.opencv.util.OpenCvUtil;

public class YunetFaceDetector extends ObjDetBasePlugin {

	private static boolean ANNOTATE_OUTPUT_IMG 		= true;
	private FaceDetectorYN faceDetectorYN = null;

	@Override
	public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		if(!isPluginOK() || aMatInput==null)
			return null;
		
		List<Mat> listOutputs = null;
		Mat srcImg 	= null;
		
		
		try {
			
	        srcImg 		= aMatInput.clone();
	        OpenCvUtil.removeAlphaChannel(srcImg);
	       
	        File fileModel 	= new File(_model_filename);
			
			int backendId 			= getDnnBackend(); 
	        int targetId 			= getDnnTarget(); 
	        int topK 				= 5000;
	                   
	        if(faceDetectorYN==null)
	        {        
		        faceDetectorYN = FaceDetectorYN.create(
		        		fileModel.getAbsolutePath(),"", 
		        		DEF_INPUT_SIZE, 
		        		(float)DEF_CONFIDENCE_THRESHOLD, 
		        		(float)DEF_NMS_THRESHOLD, 
		        		topK, backendId, targetId);
		     }
	        
	        faceDetectorYN.setInputSize(srcImg.size());
	        
	        Mat faces = new Mat();
	        faceDetectorYN.detect(srcImg,faces);
	        
	        listOutputs = new ArrayList<Mat>();
	        listOutputs.add(faces);
		}
		finally
		{
			if(srcImg!=null)
				srcImg.release();
		}
		
		return listOutputs;
	}
	
	@Override
	public Map<String,Object> parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		Map<String, Object> mapResult = new HashMap<String, Object>();
		Mat faces = aInferenceOutputMat.get(0);
		
		Mat outputImg = aMatInput.clone();
        
		
        if(faces!=null)
        {
    		FrameDetectedObj frameObjs = new FrameDetectedObj();
        	 
	        for (int i = 0; i < faces.height(); i++)
	        {
	        	int iX = (int) (faces.get(i, 0)[0]);
	        	int iY = (int) (faces.get(i, 1)[0]);
	        	int iW = (int) (faces.get(i, 2)[0]);
	        	int iH = (int) (faces.get(i, 3)[0]);
	        	
	        	Rect r = new Rect(iX, iY, iW, iH);      	
	            Imgproc.rectangle(outputImg, r, new Scalar(0, 255, 0), 2);
	            double confidence = faces.get(i, 14)[0];
	           
	            DetectedObj obj = new DetectedObj(0, "face", new Rect2d(r.x, r.y, r.width, r.height), confidence);
	            frameObjs.addDetectedObj(obj);
	            
	            if(ANNOTATE_OUTPUT_IMG)
	            {
		            Point leftEye = new Point(faces.get(i, 4)[0], faces.get(i, 5)[0]);
		            Point rightEye = new Point(faces.get(i, 6)[0], faces.get(i, 7)[0]);
		            Imgproc.circle(outputImg, leftEye, 2, new Scalar(255, 0, 0), 2); //Blue
		            Imgproc.circle(outputImg, rightEye, 2, new Scalar(0, 0, 255), 2); //Red
		            
		            Point nose = new Point(faces.get(i, 8)[0], faces.get(i, 9)[0]);
		            Imgproc.circle(outputImg, nose, 2, new Scalar(0, 255, 0), 2);
		            
		            Point leftMouth = new Point(faces.get(i, 10)[0], faces.get(i, 11)[0]);
		            Point rightMouth = new Point(faces.get(i, 12)[0], faces.get(i, 13)[0]);
		            Imgproc.circle(outputImg, leftMouth, 2, new Scalar(255, 0, 255), 2);
		            Imgproc.circle(outputImg, rightMouth, 2, new Scalar(0, 255, 255), 2);
	            }
	        }
	        
	        if(frameObjs.getTotalDetectionCount()>0)
	        {
				if(ANNOTATE_OUTPUT_IMG)
		        {
					Mat matOutputImg = DetectedObjUtil.annotateImage(aMatInput, frameObjs, null, false);
					mapResult.put(ObjDetDnnBasePlugin._KEY_OUTPUT_FRAME_ANNOTATED_IMG, matOutputImg);
		        }
		        mapResult.put(ObjDetDnnBasePlugin._KEY_OUTPUT_FRAME_DETECTIONS, frameObjs);
	        }
			//

        }
        
    	
    	return mapResult;
	}

}