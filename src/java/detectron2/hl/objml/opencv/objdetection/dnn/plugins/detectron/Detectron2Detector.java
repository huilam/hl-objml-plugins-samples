package hl.objml.opencv.objdetection.dnn.plugins.detectron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import hl.objml.opencv.objdetection.MLDetectionBasePlugin;
import hl.plugin.image.IMLDetectionPlugin;

public class Detectron2Detector extends MLDetectionBasePlugin implements IMLDetectionPlugin {
	
	private static Net NET_Detectron2 = null;
	private static List<String> OBJ_CLASSESS = new ArrayList<String>();
	
    private static final float DEF_CONFIDENCE_THRESHOLD = 0.1f;
    private static final float DEF_NMS_THRESHOLD 		= 0.4f;
    private static final Size DEF_INPUT_SIZE 			= new Size(416, 416);


	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}

	@Override
	public Map<String, Object> detect(Mat aMatInput, JSONObject aCustomThresholdJson) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		try {
			if(NET_Detectron2==null)
	        {
				init();
	        }
			
			Size sizeInput = DEF_INPUT_SIZE;
			Mat matDnnImg = aMatInput.clone();
	        
System.out.println();
System.out.println("## Loaded Image="+matDnnImg);
			
			matDnnImg = Dnn.blobFromImage(matDnnImg, 1.0/255.0, sizeInput, Scalar.all(0), true, false);
			
System.out.println("## Input blobFromImage="+matDnnImg);			
			NET_Detectron2.setInput(matDnnImg);

	        // Run inference
	        List<Mat> outputs = new ArrayList<>();
	        NET_Detectron2.forward(outputs, NET_Detectron2.getUnconnectedOutLayersNames());
	        
	        Mat matResult = outputs.get(0);
System.out.println("@@@ Inference Output="+matResult);
			
	        List<Rect2d> outputBoxes 		= new ArrayList<>();
	        List<Float> outputConfidences 	= new ArrayList<>();
	        List<Integer> outputClassIds 	= new ArrayList<>();
	        
	        float fConfidenceThreshold = DEF_CONFIDENCE_THRESHOLD;
	        float fNMSThreshold = DEF_NMS_THRESHOLD;
	        
	        decodePredictions(matResult, sizeInput, outputBoxes, outputConfidences, outputClassIds, fConfidenceThreshold);

System.out.println("@@@   Detection Boxes="+outputBoxes.size());
System.out.println("@@@   Detection Confidences="+outputConfidences.size());
System.out.println("@@@   Detection ClassIds="+outputClassIds.size());
	        
	        if(outputBoxes.size()>0)
	        {
		        MatOfInt indices = applyNMS(outputBoxes, outputConfidences, fConfidenceThreshold, fNMSThreshold);

System.out.println("## applyNMS indices="+indices);
		        
		        Mat matOutputImg = aMatInput.clone();
		        
		        // Draw bounding boxes
		        for (int idx : indices.toArray()) {
		        	
		        	System.out.println("idx="+idx);
		        	
		            Rect2d box = outputBoxes.get(idx);
		            int classId = outputClassIds.get(idx);
		            
		            String label = OBJ_CLASSESS.get(classId) + ": " + String.format("%.2f", outputConfidences.get(idx));
		            Imgproc.rectangle(matOutputImg, new Point(box.x, box.y), new Point(box.x + box.width, box.y + box.height), new Scalar(0, 255, 0), 2);
		            Imgproc.putText(matOutputImg, label, new Point(box.x, box.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
		        }
				
				mapResult.put(IMLDetectionPlugin._KEY_OUTPUT_ANNOTATED_MAT, matOutputImg);
				
	        }
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mapResult;
	}
	
	private void init()
	{
		NET_Detectron2 = Dnn.readNet( getModelFileName());
		String supporedLabels = (String) getPluginProps().get("objml.mlmodel.detection.support-labels");
		
		if(supporedLabels!=null)
		{
			String[] objs = supporedLabels.split("\n");
			OBJ_CLASSESS = new ArrayList<>(Arrays.asList(objs));
		}
	}
	
	private static MatOfInt applyNMS(List<Rect2d> aBoxesList, List<Float> aConfidencesList, float CONFIDENCE_THRESHOLD, float NMS_THRESHOLD)
	{
        // Apply Non-Maximum Suppression
        MatOfRect2d boxesMat = new MatOfRect2d();
        boxesMat.fromList(aBoxesList);
        
        MatOfFloat confidencesMat = new MatOfFloat();
        confidencesMat.fromList(aConfidencesList);
        
        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxesMat, confidencesMat, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
        return indices;

	}

	private static void decodePredictions(
	        Mat output, Size imageSize, 
	        List<Rect2d> boxes, List<Float> confidences, List<Integer> classIds,
	        float CONFIDENCE_THRESHOLD) {

	}
}