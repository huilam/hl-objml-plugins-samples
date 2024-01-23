package hl.objml.opencv.objdetection.dnn.plugins.yunet.face;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import org.json.JSONObject;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;

import hl.objml.opencv.objdetection.MLDetectionBasePlugin;
import hl.opencv.util.OpenCvUtil;
import hl.plugin.image.IMLDetectionPlugin;

public class FaceDetector extends MLDetectionBasePlugin implements IMLDetectionPlugin {

	private FaceDetectorYN faceDetectorYN = null;
	
	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}

	@Override
	public Properties getPluginProps() {
		return super.getPluginProps();
	}
	
	@Override
	public String getPluginName() {
		return super.getModelName();
	}
	
	@Override
	public String getPluginMLModelFileName() {
		return super.getModelFileName();
	}
	
	@Override
	public Map<String, Object> detect(Mat aMatInput, JSONObject aCustomThresholdJson) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		try {
			Mat matOutput = faceDetect(aMatInput);
			if(matOutput!=null && !matOutput.empty())
			{
				mapResult.put(IMLDetectionPlugin._KEY_MAT_OUTPUT, matOutput);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mapResult;
	}
	
	/////
	private Mat faceDetect(Mat aMatInput) throws Exception{
		
		if(!isPluginOK() || aMatInput==null)
			return null;
		
		Mat srcImg 	= null;
		Mat faces 	= null;
		
		try {
	        srcImg 		= aMatInput.clone();
	        OpenCvUtil.removeAlphaChannel(srcImg);
	       
	        long lStartMs 	= 0;
	        
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
	            lStartMs = System.currentTimeMillis();
		        faceDetectorYN = FaceDetectorYN.create(
		        		fileModel.getAbsolutePath(),"", 
		        		new Size(320, 320),scoreThreshold, 
		        		nmsThreshold, topK, backendId, targetId);
	//	        System.out.println();
	//	        System.out.println(" - "+fileModel.getName()+" loaded : "+(System.currentTimeMillis()-lStartMs)+"ms");
		     }
	        
	        
	        lStartMs = System.currentTimeMillis();
	        faceDetectorYN.setInputSize(srcImg.size());
	        
	        faces = new Mat();
	        faceDetectorYN.detect(srcImg,faces);
	        System.out.println(" - Detected face = "+faces.height()+" : "+(System.currentTimeMillis()-lStartMs)+"ms");
	        for (int i = 0; i < faces.height(); i++)
	        {
	        	Rect r = new Rect((int) (faces.get(i, 0)[0]), (int)(faces.get(i, 1)[0]), (int)(faces.get(i, 2)[0]), (int)(faces.get(i, 3)[0]));      	
	            Imgproc.rectangle(srcImg, r, new Scalar(0, 255, 0), 2);
	
	            Imgproc.circle(srcImg, new Point(faces.get(i, 4)[0], faces.get(i, 5)[0]), 2,new Scalar(255, 0, 0), 2);
	            Imgproc.circle(srcImg, new Point(faces.get(i, 6)[0], faces.get(i, 7)[0]), 2, new Scalar(0, 0, 255), 2);
	            Imgproc.circle(srcImg, new Point(faces.get(i, 8)[0], faces.get(i, 9)[0]), 2, new Scalar(0, 255, 0), 2);
	            Imgproc.circle(srcImg, new Point(faces.get(i, 10)[0], faces.get(i, 11)[0]), 2, new Scalar(255, 0, 255), 2);
	            Imgproc.circle(srcImg, new Point(faces.get(i, 12)[0], faces.get(i, 13)[0]), 2, new Scalar(0, 255, 255), 2);
	        }
	        
	        if(faces.height()<=0)
	        {
	        	if(srcImg!=null)
	        	{
	        		srcImg.release();
	        	}
	        	srcImg = null;
	        }
		}finally
		{
			
			if(faces!=null)
				faces.release();
		}
		return srcImg;
    }

}