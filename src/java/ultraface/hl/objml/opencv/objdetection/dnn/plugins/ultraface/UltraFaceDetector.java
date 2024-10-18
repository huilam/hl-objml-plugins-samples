package hl.objml.opencv.objdetection.dnn.plugins.ultraface;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.ObjDetDnnBasePlugin;


public class UltraFaceDetector extends ObjDetDnnBasePlugin {
	
    private static boolean SWAP_RB_CHANNEL			= true;
    private static boolean APPLY_IMG_PADDING 		= false;
    private static boolean RESIZE_INPUT_IMAGE 		= false;
    private static boolean ANNOTATE_OUTPUT_IMG 		= true;


	/**
	 *  WIP
	 *  
	 *  https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
	 *  
	 *  ONNX Model = https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/models/onnx
	 *  
	 */
	@Override
	public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		if(aDnnNet==null)
			aDnnNet = getDnnNet();
		
		Mat matDnnImg 		= null;
		List<Mat> outputs 	= null;
		try {
			// Prepare input
			Size sizeInput 	= DEF_INPUT_SIZE;
			matDnnImg = aMatInput.clone();					
			matDnnImg 	= inferencePreProcess(matDnnImg, sizeInput, APPLY_IMG_PADDING, SWAP_RB_CHANNEL);
			aDnnNet.setInput(matDnnImg);
			
	        // Run the forward pass
	        outputs = new ArrayList<>();
	        List<String> outNames = new ArrayList<>();
	        outNames.add("boxes");   // Name of the bounding boxes output
	        outNames.add("scores");  // Name of the scores output
	        aDnnNet.forward(outputs, outNames);
		}
		finally
		{
			if(matDnnImg!=null)
				matDnnImg.release();
		}
        return outputs;
	}
	
	@Override
	public Map<String,Object> parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		Map<String, Object> mapResult = new HashMap<String, Object>();
		try {

			 // Decode detection
	        double dConfThreshold 			= super.DEF_CONFIDENCE_THRESHOLD;
	        double dNMSThreshold 			= super.DEF_NMS_THRESHOLD;
	        
	        List<Rect2d> outputBoxes 		= new ArrayList<>();
	        List<Float> outputConfidences 	= new ArrayList<>();
	        List<Integer> outputClassIds 	= new ArrayList<>();
	        //
	        decodePredictions(aInferenceOutputMat, 
	        		aMatInput.size(),  
	        		outputBoxes, outputConfidences, outputClassIds, 
	        		dConfThreshold);
	        //
	        FrameDetectedObj frameObjs = new FrameDetectedObj();
	        if(outputBoxes.size()>0)
	        {
	        	 // Apply NMS
		        int[] indices = applyNMS(outputBoxes, outputConfidences, dConfThreshold, dNMSThreshold);

		        // Calculate bounding boxes
		        for (int idx : indices) {
		        	
		            Rect2d box 			= outputBoxes.get(idx);
		            int classId 		= outputClassIds.get(idx);
		            String classLabel 	= OBJ_CLASSESS.get(classId);
		            Float confScore 	= outputConfidences.get(idx);
		            
		            DetectedObj obj = new DetectedObj(classId, classLabel, box, confScore);
		            frameObjs.addDetectedObj(obj);
		        }
		        
		        // Draw bounding boxes
				if(ANNOTATE_OUTPUT_IMG)
		        {
					Mat matOutputImg = DetectedObjUtil.annotateImage(aMatInput, frameObjs, null, false);
					mapResult.put(ObjDetDnnBasePlugin._KEY_OUTPUT_FRAME_ANNOTATED_IMG, matOutputImg);
		        }
		        mapResult.put(ObjDetDnnBasePlugin._KEY_OUTPUT_FRAME_DETECTIONS, frameObjs);
		     }
	        
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mapResult;	
	}
	
	private static int[] applyNMS(List<Rect2d> aBoxesList, List<Float> aConfidencesList, 
			double CONFIDENCE_THRESHOLD, double NMS_THRESHOLD)
	{
        MatOfInt indices = new MatOfInt();

        if(aBoxesList.size()>0)
        {
	        // Apply Non-Maximum Suppression
	        MatOfRect2d boxesMat = new MatOfRect2d();
	        boxesMat.fromList(aBoxesList);
	        
	        MatOfFloat confidencesMat = new MatOfFloat();
	        confidencesMat.fromList(aConfidencesList);
	        
	        Dnn.NMSBoxes(boxesMat, confidencesMat, (float)CONFIDENCE_THRESHOLD, (float)NMS_THRESHOLD, indices);
        }
        return indices.toArray();

	}
	
	private static Mat inferencePreProcess(Mat aMatInput, Size sizeInput, 
			boolean isApplyImgPadding, boolean isSwapRBChannel)
	{
		if(isApplyImgPadding)
		{
			Mat matPaddedImg = null;
			Mat matROI = null;
			try {
				
				int iMaxPixels = Math.max(aMatInput.width(), aMatInput.height());
				matPaddedImg = new Mat(new Size(iMaxPixels,iMaxPixels), aMatInput.type(), Scalar.all(127));
				matROI = matPaddedImg.submat(0,aMatInput.rows(),0,aMatInput.cols());
				aMatInput.copyTo(matROI);
				aMatInput = matPaddedImg.clone();
			}
			finally
			{
				if(matPaddedImg!=null)
					matPaddedImg.release();
				if(matROI!=null)
					matROI.release();
			}
		}
		
		//resize
		if(RESIZE_INPUT_IMAGE)
		{
			Imgproc.resize(aMatInput, aMatInput, sizeInput);
		}
		
		// Convert from BGR to RGB
		if(isSwapRBChannel)
		{
			Imgproc.cvtColor(aMatInput, aMatInput, Imgproc.COLOR_BGR2RGB);
		}

		return Dnn.blobFromImage(aMatInput, 1.0/128, sizeInput, Scalar.all(127), true, false);		
	}

	private void decodePredictions(
	        List<Mat> matOutputs, 
	        Size sizeOrg,
	        List<Rect2d> boxes, List<Float> confidences, List<Integer> classIds,
	        final double aConfidenceThreshold) {
	    
		// https://docs.openvino.ai/2024/omz_models_model_ultra_lightweight_face_detection_rfb_320.html
		// Input Mat = 1 * 3(Color) * 240(H) * 320(W)
		// Output Mat = Box:{1, 4420, 4} , Score {1, 4420, 2}
		//
		// for 640 model input 640x480, output box {1,17640,4}
		
		
        Mat matBoxes 	=  matOutputs.get(0);
        Mat matScores 	=  matOutputs.get(1);
        
        int totalAnchors = matBoxes.size(1);
        
        matBoxes 	= matBoxes.reshape(1, new int[] {totalAnchors, 4});
        matScores 	= matScores.reshape(1, new int[] {totalAnchors, 2});
        
		double dScaleW = sizeOrg.width / DEF_INPUT_SIZE.width;
		double dScaleH = sizeOrg.height /DEF_INPUT_SIZE.height;
		
		for (int i = 0; i < matBoxes.rows(); i++) {
			
			double[] dScores = matScores.get(i, 1);
		    double dConfScore = dScores[0];
		    if(dConfScore > aConfidenceThreshold)
		    {
			    double left 	= matBoxes.get(i, 0)[0] * DEF_INPUT_SIZE.width;
			    double top 		= matBoxes.get(i, 1)[0] * DEF_INPUT_SIZE.height;
			    //
			    double right 	= matBoxes.get(i, 2)[0] * DEF_INPUT_SIZE.width;
			    double bottom 	= matBoxes.get(i, 3)[0] * DEF_INPUT_SIZE.height;
			
			    // Calculate width and height
			    double width 	= right - left;
			    double height 	= bottom - top;
			    
			    left *= dScaleW;
			    width *= dScaleW;
			    
			    top *= dScaleH;
			    height *= dScaleH;
			    
			    // Create a new Rect2d object
			    Rect2d rect = new Rect2d(left, top, width, height);		    
			    boxes.add(rect);
			
			    // Add the confidence score
			    confidences.add((float)dConfScore);
			    classIds.add(0);
		    }
		}
	}
}