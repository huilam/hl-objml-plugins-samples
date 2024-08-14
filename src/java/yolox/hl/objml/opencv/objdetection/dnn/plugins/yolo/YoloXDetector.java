package hl.objml.opencv.objdetection.dnn.plugins.yolo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Core;
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
	
	private static Net NET_YOLOX = null;
	
	private static final int[] STRIDES =  {8, 16, 32};
	private static List<String> OBJ_CLASSESS = new ArrayList<String>();
	
    private static float DEF_CONFIDENCE_THRESHOLD = 0.5f;
    private static float DEF_NMS_THRESHOLD 		= 0.4f;
    private static Size DEF_INPUT_SIZE 			= new Size(640, 640);


	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}

	@Override
	public Map<String, Object> detect(Mat aMatInput, JSONObject aCustomThresholdJson) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		try {
			if(NET_YOLOX==null)
	        {
				init();
	        }
			
			Size sizeInput = DEF_INPUT_SIZE;
			Mat matDnnImg = aMatInput.clone();
			
			 // Convert from BGR to RGB
	        Imgproc.cvtColor(matDnnImg, matDnnImg, Imgproc.COLOR_BGR2RGB);
	        
System.out.println();
System.out.println("## Loaded Image="+matDnnImg);
			
			matDnnImg = Dnn.blobFromImage(matDnnImg, 1.0 / 255.0, sizeInput, Scalar.all(0), true, false);
			NET_YOLOX.setInput(matDnnImg);
System.out.println("## Dnn Input Image="+matDnnImg);

	        // Run inference
	        List<Mat> outputs = new ArrayList<>();
	        NET_YOLOX.forward(outputs, NET_YOLOX.getUnconnectedOutLayersNames());
	        
	        Mat matResult = outputs.get(0);
System.out.println("@@@ Inference Output="+matResult);
			
			matResult = postProcess(matResult, sizeInput);

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
		        int[] indices = applyNMS(outputBoxes, outputConfidences, fConfidenceThreshold, fNMSThreshold);

System.out.println("## applyNMS indices.length="+indices.length);
		        
		        Mat matOutputImg = aMatInput.clone();
		        
		        // Draw bounding boxes
		        for (int idx : indices) {
		        	
		            Rect2d box = outputBoxes.get(idx);
		            int classId = outputClassIds.get(idx);
		            
		            String label = OBJ_CLASSESS.get(classId) + ": " + String.format("%.2f", outputConfidences.get(idx));
		            
System.out.println("idx="+label+" "+box.tl()+" "+box.br());		            
		            Imgproc.rectangle(matOutputImg, new Point(box.x, box.y), new Point(box.x + box.width, box.y + box.height), new Scalar(0, 255, 0), 2);
		            Imgproc.putText(matOutputImg, label, new Point(box.x, box.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
		        }
				
				mapResult.put(IMLDetectionPlugin._KEY_MAT_OUTPUT, matOutputImg);
				
	        }
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mapResult;
	}
	
	private void init()
	{
		NET_YOLOX = Dnn.readNet( getModelFileName());
		
		if(NET_YOLOX!=null)
		{
			String sSupporedLabels = (String) getPluginProps().get("objml.mlmodel.detection.support-labels");
			if(sSupporedLabels!=null)
			{
				String[] objs = sSupporedLabels.split("\n");
				OBJ_CLASSESS = new ArrayList<>(Arrays.asList(objs));
			}
			//
			String sConfThreshold = (String) getPluginProps().get("objml.mlmodel.detection.confidence-threshold");
			if(sConfThreshold!=null)
			{
				try {
					DEF_CONFIDENCE_THRESHOLD = Float.parseFloat(sConfThreshold);
				}catch(NumberFormatException ex)
				{
					ex.printStackTrace();
				}
			}
			//
			String sNMSThreshold = (String) getPluginProps().get("objml.mlmodel.detection.nms-threshold");
			if(sNMSThreshold!=null)
			{
				try {
					DEF_NMS_THRESHOLD = Float.parseFloat(sNMSThreshold);
				}catch(NumberFormatException ex)
				{
					ex.printStackTrace();
				}
			}
			//
			String sInputImageSize = (String) getPluginProps().get("objml.mlmodel.detection.input-size");
			if(sInputImageSize!=null)
			{

				String sSeparator = "x";
				if(sInputImageSize.indexOf(sSeparator)==-1)
					sSeparator = ",";
				
				double dWidth = 0;
				double dHeight = 0;
				String[] sSize = sInputImageSize.split(sSeparator);
				if(sSize.length>0)
				{
					try {
						dWidth 	= Double.parseDouble(sSize[0]);
						dHeight = dWidth;
						if(sSize.length>1)
						{
							dHeight = Double.parseDouble(sSize[1]);
						}
					}
					catch(NumberFormatException ex)
					{
						ex.printStackTrace();
					}
					DEF_INPUT_SIZE = new Size(dWidth,dHeight);
				}
						
				}
			//System.out.println();
			//System.out.println("*init* DEF_CONFIDENCE_THRESHOLD="+DEF_CONFIDENCE_THRESHOLD);
			//System.out.println("*init* DEF_NMS_THRESHOLD="+DEF_NMS_THRESHOLD);
			//System.out.println("*init* DEF_INPUT_SIZE="+DEF_INPUT_SIZE);

		}
	}
	
	private static Mat postProcess(Mat matOutputDetections, Size inputSize)
	{
		//https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py
			
		int detectionCount = 0;
		int[] hSizes = new int[STRIDES.length];
		int[] wSizes = new int[STRIDES.length];
		
		for (int i=0; i<STRIDES.length; i++) {
			hSizes[i] = Math.floorDiv((int)inputSize.height, STRIDES[i]);
			wSizes[i] = Math.floorDiv((int)inputSize.width, STRIDES[i]);
			detectionCount += hSizes[i]*wSizes[i];
		}
		
		int[] matOutputDetectionsShape = {8400, 85};
		matOutputDetections = matOutputDetections.reshape(1, matOutputDetectionsShape);

		
		int iOutputDetectionSize = matOutputDetections.size(0);
		
		if (detectionCount != iOutputDetectionSize) {
			System.err.println("The ML model output is not as expected ! detectionCount:"+detectionCount+" != iOutputDetectionSize:"+iOutputDetectionSize);
			return null;
		}
		///////////////////////

		int detectionIdx = 0;
		for (int i = 0; i < STRIDES.length; i++) {
            int hSize = hSizes[i];
            int wSize = wSizes[i];
            int stride = STRIDES[i];

            for (int y = 0; y < hSize; y++) {
                for (int x = 0; x < wSize; x++) {

                    double value0 = matOutputDetections.get(detectionIdx, 0)[0];
                    value0 = (value0 + x) * stride;
                    matOutputDetections.put(detectionIdx, 0, value0);

                    double value1 = matOutputDetections.get(detectionIdx, 1)[0];
                    value1 = (value1 + y) * stride;
                    matOutputDetections.put(detectionIdx, 1, value1);

                    double value2 = matOutputDetections.get(detectionIdx, 2)[0];
                    value2 = Math.exp(value2) * stride;
                    matOutputDetections.put(detectionIdx, 2, value2);

                    double value3 = matOutputDetections.get(detectionIdx, 3)[0];
                    value3 = Math.exp(value3) * stride;
                    matOutputDetections.put(detectionIdx, 3, value3);
                    
                    detectionIdx++;
                }
            }
        }
        
        return matOutputDetections;
    }

	
	private static int[] applyNMS(List<Rect2d> aBoxesList, List<Float> aConfidencesList, float CONFIDENCE_THRESHOLD, float NMS_THRESHOLD)
	{
        // Apply Non-Maximum Suppression
        MatOfRect2d boxesMat = new MatOfRect2d();
        boxesMat.fromList(aBoxesList);
        
        MatOfFloat confidencesMat = new MatOfFloat();
        confidencesMat.fromList(aConfidencesList);
        
        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxesMat, confidencesMat, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
        
        return indices.toArray();

	}

	private static void decodePredictions(
	        Mat matResult, Size imageSize, 
	        List<Rect2d> boxes, List<Float> confidences, List<Integer> classIds,
	        float CONFIDENCE_THRESHOLD) {

	    int iImgW = (int) imageSize.width;
	    int iImgH = (int) imageSize.height;
        for (int i = 0; i < matResult.rows(); i++) {
            Mat row = matResult.row(i);
            Mat scores = row.colRange(5, matResult.cols());
            Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
            float confidence = (float) mm.maxVal;
            
            if (confidence > CONFIDENCE_THRESHOLD) {
                int classId = (int) mm.maxLoc.x;
                float[] data = new float[4];
                row.colRange(0, 4).get(0, 0, data);

                int centerX = (int) (data[0]);
                int centerY = (int) (data[1]);
                int width 	= (int) (data[2]);
                int height 	= (int) (data[3]);
                
                int left 	= centerX - width / 2;
                int top 	= centerY - height / 2;
                
                top = top<0? 0: top;
                left = left<0? 0: left;
                width = left+width<iImgW? width: iImgW-left-1;
        		height = top+height<iImgH? height: iImgH-top-1;

                classIds.add(classId);
                confidences.add(confidence);
                boxes.add(new Rect2d(left, top, width, height));
            }
        }
	}
}