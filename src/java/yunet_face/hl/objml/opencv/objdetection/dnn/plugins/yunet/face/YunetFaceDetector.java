package hl.objml.opencv.objdetection.dnn.plugins.yunet.face;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;

import hl.objml.opencv.objdetection.MLDetectionBasePlugin;
import hl.opencv.util.OpenCvUtil;
import hl.plugin.image.IMLDetectionPlugin;
import hl.plugin.image.ObjDetection;

public class YunetFaceDetector extends MLDetectionBasePlugin {

	private FaceDetectorYN faceDetectorYN = null;
	
	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}

	@Override
	public Map<String, Object> detect(Mat aMatInput, JSONObject aCustomThresholdJson) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		
		if(aMatInput!=null)
		{
			mapResult = faceDetect(aMatInput);
		}
		
		return mapResult;
	}
	
	/////
	private Map<String, Object>  faceDetect(Mat aMatInput) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		
		if(!isPluginOK() || aMatInput==null)
			return null;
		
		Mat srcImg 	= null;
		Mat faces 	= null;
		
		
		try {
	        srcImg 		= aMatInput.clone();
	        OpenCvUtil.removeAlphaChannel(srcImg);
	       
	        File fileModel 	= new File(_model_filename);
			
	        // 0: default, 1: Halide, 2: Intel's Inference Engine, 3: OpenCV, 4: VKCOM, 5: CUDA
			int backendId 			= 0; 
			// 0: CPU, 1: OpenCL, 2: OpenCL FP16, 3: Myriad, 4: Vulkan, 5: FPGA, 6: CUDA, 7: CUDA FP16, 8: HDDL
	        int targetId 			= 0; 
	        float scoreThreshold 	= 0.9f;
	        float nmsThreshold 		= 0.3f;
	        int topK 				= 5000;
	                   
	        if(faceDetectorYN==null)
	        {        
		        faceDetectorYN = FaceDetectorYN.create(
		        		fileModel.getAbsolutePath(),"", 
		        		new Size(320, 320),scoreThreshold, 
		        		nmsThreshold, topK, backendId, targetId);
		     }
	        
	        faceDetectorYN.setInputSize(srcImg.size());
	        
	        faces = new Mat();
	        faceDetectorYN.detect(srcImg,faces);
	        
	        if(faces!=null)
	        {
	        	ObjDetection objs = new ObjDetection();
	        	 
		        for (int i = 0; i < faces.height(); i++)
		        {
		        	int iX = (int) (faces.get(i, 0)[0]);
		        	int iY = (int) (faces.get(i, 1)[0]);
		        	int iW = (int) (faces.get(i, 2)[0]);
		        	int iH = (int) (faces.get(i, 3)[0]);
		        	
		        	Rect r = new Rect(iX, iY, iW, iH);      	
		            Imgproc.rectangle(srcImg, r, new Scalar(0, 255, 0), 2);
		            double confidence = faces.get(i, 14)[0];
		           
		            objs.addDetectedObj(0, "face", confidence, new Rect2d(r.x, r.y, r.width, r.height));
		
		            Point leftEye = new Point(faces.get(i, 4)[0], faces.get(i, 5)[0]);
		            Point rightEye = new Point(faces.get(i, 6)[0], faces.get(i, 7)[0]);
		            Imgproc.circle(srcImg, leftEye, 2, new Scalar(255, 0, 0), 2); //Blue
		            Imgproc.circle(srcImg, rightEye, 2, new Scalar(0, 0, 255), 2); //Red
		            
		            Point nose = new Point(faces.get(i, 8)[0], faces.get(i, 9)[0]);
		            Imgproc.circle(srcImg, nose, 2, new Scalar(0, 255, 0), 2);
		            
		            
		            Point leftMouth = new Point(faces.get(i, 10)[0], faces.get(i, 11)[0]);
		            Point rightMouth = new Point(faces.get(i, 12)[0], faces.get(i, 13)[0]);
		            Imgproc.circle(srcImg, leftMouth, 2, new Scalar(255, 0, 255), 2);
		            Imgproc.circle(srcImg, rightMouth, 2, new Scalar(0, 255, 255), 2);
		        }
	        
	        	mapResult.put(IMLDetectionPlugin._KEY_THRESHOLD_DETECTION, objs.toJson());
	        	mapResult.put(IMLDetectionPlugin._KEY_OUTPUT_TOTAL_COUNT, faces.height());
	        }
	        
	        if(srcImg!=null)
	        {
	        	mapResult.put(IMLDetectionPlugin._KEY_OUTPUT_ANNOTATED_MAT, srcImg.clone());
	        }
	        
	    	mapResult.put(IMLDetectionPlugin._KEY_THRESHOLD_DETECTION, scoreThreshold);
			mapResult.put(IMLDetectionPlugin._KEY_THRESHOLD_NMS, nmsThreshold);
	        
		}finally
		{
			
			if(faces!=null)
			{
				faces.release();
			}
			
			if(srcImg!=null)
        	{
        		srcImg.release();
        	}
		}
		return mapResult;
    }

}