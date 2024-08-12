package hl.objml.opencv.objdetection.dnn.plugins.yolox;

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

public class YoloXDetector extends MLDetectionBasePlugin implements IMLDetectionPlugin {
	
	private static Net netYoloX = null;
	private static Size sizeInput = new Size(640,480);
	private static List<String> objClasses = new ArrayList<String>();


	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}

	@Override
	public Map<String, Object> detect(Mat aMatInput, JSONObject aCustomThresholdJson) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		try {
			if(netYoloX==null)
	        {
				init();
	        }
			
System.out.println("aMatInput="+aMatInput);
			
			
			Mat matDnnImg = Dnn.blobFromImage(aMatInput.clone(), 1.0 / 255.0, sizeInput, new Scalar(0, 0, 0), true, false);
	        netYoloX.setInput(matDnnImg);

	        // Run inference
	        List<Mat> outputs = new ArrayList<>();
	        netYoloX.forward(outputs, netYoloX.getUnconnectedOutLayersNames());

	        List<Rect2d> outputBoxes = new ArrayList<>();
	        List<Float> outputConfidences = new ArrayList<>();
	        List<Integer> outputClassIds = new ArrayList<>();
	        
	        
	        decodePredictions(
	        		0.5f,
	        		outputs.get(0), aMatInput.size(), outputBoxes, outputConfidences, outputClassIds);

	        MatOfInt indices = applyNMS(outputBoxes, outputConfidences, 0.5f, 0.4f);
	        
	        Mat matOutput = aMatInput.clone();
	        
	        // Draw bounding boxes
	        for (int idx : indices.toArray()) {
	            Rect2d box = outputBoxes.get(idx);
	            int classId = outputClassIds.get(idx);
	            String label = objClasses.get(classId) + ": " + String.format("%.2f", outputConfidences.get(idx));
	            Imgproc.rectangle(matOutput, new Point(box.x, box.y), new Point(box.x + box.width, box.y + box.height), new Scalar(0, 255, 0), 2);
	            Imgproc.putText(matOutput, label, new Point(box.x, box.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
	        }
			
			mapResult.put(IMLDetectionPlugin._KEY_MAT_OUTPUT, matOutput);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mapResult;
	}
	
	private void init()
	{
		netYoloX = Dnn.readNetFromONNX( getModelFileName());
		String supporedLabels = (String) getPluginProps().get("objml.mlmodel.detection.support-labels");
		
		if(supporedLabels!=null)
		{
			String[] objs = supporedLabels.split("\n");
			objClasses = new ArrayList<>(Arrays.asList(objs));
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
    		float CONFIDENCE_THRESHOLD,
    		Mat output, Size imageSize, 
    		List<Rect2d> boxes, List<Float> confidences, List<Integer> classIds) {
        int width = (int) imageSize.width;
        int height = (int) imageSize.height;

        for (int i = 0; i < output.rows(); i++) {
            Mat row = output.row(i);
            float[] data = new float[(int) row.total()];
            row.get(0, 0, data);

            float confidence = data[4];
            if (confidence > CONFIDENCE_THRESHOLD) {
                int centerX = (int) (data[0] * width);
                int centerY = (int) (data[1] * height);
                int w = (int) (data[2] * width);
                int h = (int) (data[3] * height);

                int x = centerX - w / 2;
                int y = centerY - h / 2;

                boxes.add(new Rect2d(x, y, w, h));
                confidences.add(confidence);
                classIds.add(getClassId(data));
            }
        }
    }
    
    private static int getClassId(float[] data) {
        int startIndex = 5;  // assuming the class scores start after the bounding box coordinates and confidence
        float maxScore = Float.MIN_VALUE;
        int classId = -1;

        for (int i = startIndex; i < data.length; i++) {
            if (data[i] > maxScore) {
                maxScore = data[i];
                classId = i - startIndex;
            }
        }
        return classId;
    }

}