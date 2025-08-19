package hl.objml.opencv.objdetection.dnn.plugins.yolov11;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
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


public class YoloV11Detector extends ObjDetDnnBasePlugin {
	
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
        //
        decodePredictions(matResult, 
        		scaleOrgW, scaleOrgH,  
        		outputBoxes, outputConfidences, outputClassIds, 
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
	        final double aConfidenceThreshold) {
		
System.out.println("decodePredictions-matResult="+matResult);
		// matResult=Mat [ 1*84*8400*CV_32FC1]

		int objClassInfo = matResult.size(1);
		int totalAnchors = matResult.size(2);
		
		matResult = matResult.reshape(1, new int[] {objClassInfo, totalAnchors});
		Core.transpose(matResult, matResult); //swap
		
		int iEndCol = matResult.cols();
		if(matResult.cols()>getTotalObjClsCount()+4)
		{
			iEndCol -= 32;
		}
	    
		for (int i = 0; i < matResult.rows(); i++) 
		{
		    Mat row = matResult.row(i);
		    Mat scores = row.colRange(4, iEndCol);
		    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
		    float confidence = (float) mm.maxVal;
		    
		    if (confidence >= aConfidenceThreshold) {
		        int classId = (int) mm.maxLoc.x;
		        
		        //check if obj of interest
				if(isObjOfInterest(classId))
				{
					float[] data = new float[4];
		            row.colRange(0, 4).get(0, 0, data);
		
		            double centerX = data[0] * aScaleW;
		            double centerY = data[1] * aScaleH;
		            double width   = data[2] * aScaleW;
		            double height  = data[3] * aScaleH;
		            
		            double left = centerX - (width / 2);
		            double top 	= centerY - (height / 2);
		            
		            long lLeft  	= (long) Math.floor((left<0? 0: left));
		            long lTop  		= (long) Math.floor((top<0? 0: top));
		            long lWidth 	= (long) Math.floor(width);
		    		long lHeight 	= (long) Math.floor(height);
		    		
		            classIds.add(classId);
		            confidences.add(confidence);
		            boxes.add(new Rect2d(lLeft , lTop, lWidth, lHeight));
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