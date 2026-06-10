package hl.objml.opencv.doclayout.dnn.plugins.yolo;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import hl.objml2.common.DetectedObj;
import hl.objml2.common.DetectedObjUtil;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.MLPluginFrameOutput;
import hl.objml2.plugin.ObjDetDnnBasePlugin;


public class DocLayoutYolo10Detector extends ObjDetDnnBasePlugin {
	
    private static boolean ANNOTATE_OUTPUT_IMG 	= true;
    
    private int total_obj_count 		= -1;

    public int getTotalObjClsCount()
    {
    	if(total_obj_count <0)
    	{
    		total_obj_count = getSupportedObjLabels().length;
    	}
    	return total_obj_count;
    }
	/**
	 *
	 */
	@Override
    public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		List<Mat> outputs = null;
		Mat matInputImg = null;
		Mat matDnnImg = null;
		try {
			
			// Prepare input					
			matDnnImg = Dnn.blobFromImage(aMatInput, 
					1d / 255d, getImageInputSize(), 
					Scalar.all(0), true, false);
			aDnnNet.setInput(matDnnImg);

	        // Run inference
	        outputs = new ArrayList<>();
	        aDnnNet.forward(outputs, aDnnNet.getUnconnectedOutLayersNames());
		}
		finally
		{
			if(matInputImg!=null)
				matInputImg.release();
			
			if(matDnnImg!=null)
				matDnnImg.release();
		}
		return outputs;
			
	}
	
	@Override
    public MLPluginFrameOutput parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		MLPluginFrameOutput frameOutput = new MLPluginFrameOutput();
		
		int m = 0;
		for(Mat matOutput : aInferenceOutputMat)
		{
			System.out.println(" "+m+" = "+matOutput);
			m++;
		}
		
		Mat matResult = aInferenceOutputMat.get(0);
		
		 // Decode detection
        double scaleOrgW = aMatInput.width() / getImageInputSize().width;
        double scaleOrgH = aMatInput.height() / getImageInputSize().height;
        
        double fConfidenceThreshold 	= getConfidenceThreshold();
        double fNMSThreshold 			= getNMSThreshold();
        
        List<Rect2d> outputBoxes 		= new ArrayList<>();
        List<Float> outputConfidences 	= new ArrayList<>();
        List<Integer> outputClassIds 	= new ArrayList<>();
        List<float[]> outputMask		= new ArrayList<>();
        //
        decodePredictions(matResult, 
        		scaleOrgW, scaleOrgH,  
        		outputBoxes, outputConfidences, outputClassIds, outputMask,
        		fConfidenceThreshold);
        //
        if(outputBoxes.size()>0)
        {
	        // Apply NMS
	        int[] indices = applyNMS(outputBoxes, outputConfidences, fConfidenceThreshold, fNMSThreshold);

	        // Calculate bounding boxes
	        FrameDetectedObj frameObjs = new FrameDetectedObj();
	        for (int idx : indices) {
	        	
	            Rect2d box 			= outputBoxes.get(idx);
	            int classId 		= outputClassIds.get(idx);
	            String classLabel 	= getObjClassLabel(classId);
	            Float confScore 	= outputConfidences.get(idx);
	            
	            if(outputMask.size()>idx)
	            {
	            	//TODO: Pending decode of Mask
	            	//float[] maskCoeffs 	= outputMask.get(idx);
	                //Mat coeffMat = new Mat(1, 32, CvType.CV_32F);
	                //coeffMat.put(0, 0, maskCoeffs);
	            }
	            
	            DetectedObj obj = new DetectedObj(classId, classLabel, box, confScore);
	            frameObjs.addDetectedObj(obj);
	        }
	        
	        // Draw bounding boxes
			if(ANNOTATE_OUTPUT_IMG)
	        {
				Mat matOutputImg = DetectedObjUtil.annotateImage(aMatInput, frameObjs, null, false);
				frameOutput.setAnnotatedFrameImage(matOutputImg);
	        }
			frameOutput.setFrameDetectedObj(frameObjs);
			//
        }
        
        return frameOutput;
	}
	
	private void decodePredictions(
	        Mat matResult, 
	        final double aScaleW,
	        final double aScaleH,
	        List<Rect2d> boxes, List<Float> confidences, List<Integer> classIds,
	        List<float[]> outputMask,
	        final double aConfidenceThreshold) {

	    // DocLayout-YOLO (YOLOv10) outputs [1, 300, 6]
	    // 300 = max boxes, 6 = [x1, y1, x2, y2, confidence, classId]
	    
	    int numBoxes = matResult.size(1); // 300
	    
	    // Reshape from 3D [1, 300, 6] to 2D [300, 6]
	    matResult = matResult.reshape(1, numBoxes);
	    
	    float[] data = new float[6];

	    for (int i = 0; i < matResult.rows(); i++) {
	        matResult.get(i, 0, data); // Pull all 6 values for this detection row
	        
	        float confidence = data[4];
	        
	        if (confidence >= aConfidenceThreshold) {
	            int classId = (int) data[5];
	            
	            if (isObjOfInterest(classId)) {
	                // YOLOv10 outputs absolute coordinates directly
	                double left   = data[0] * aScaleW;
	                double top    = data[1] * aScaleH;
	                double right  = data[2] * aScaleW;
	                double bottom = data[3] * aScaleH;
	                
	                double width  = right - left;
	                double height = bottom - top;
	                
	                long lLeft    = (long) Math.max(0, left);
	                long lTop     = (long) Math.max(0, top);
	                long lWidth   = (long) Math.max(0, width);
	                long lHeight  = (long) Math.max(0, height);
	                
	                classIds.add(classId);
	                confidences.add(confidence);
	                boxes.add(new Rect2d(lLeft, lTop, lWidth, lHeight));
	                
	                // Note: DocLayout-YOLO typically does not output segmentation masks.
	                // If you ever use a custom variant that does, append to outputMask here.
	            }
	        }
	    }
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

}