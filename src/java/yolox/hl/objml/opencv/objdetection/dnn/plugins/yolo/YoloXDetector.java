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
import hl.opencv.util.OpenCvUtil;
import hl.plugin.image.IMLDetectionPlugin;

public class YoloXDetector extends MLDetectionBasePlugin implements IMLDetectionPlugin {
	
	private static final int[] STRIDES 	=  {8, 16, 32};
	
	private static Net NET_YOLOX 					= null;
	private static List<String> OBJ_CLASSESS 		= new ArrayList<String>();
    private static float DEF_CONFIDENCE_THRESHOLD 	= 0.9f;
    private static float DEF_NMS_THRESHOLD 			= 0.8f;
    private static Size DEF_INPUT_SIZE 				= new Size(640, 640);
    
    private static boolean SWAP_RB_CHANNEL			= false;
    private static boolean APPLY_IMG_PADDING 		= true;


	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}

	/**
	 *  ONNX Model = https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime
	 *  Processing Reference
	 *  - https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py
	 *  - https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py
	 *  - https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py
	 */
	public Map<String, Object> detect(Mat aMatInput, JSONObject aCustomThresholdJson) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		try {
			if(NET_YOLOX==null)
	        {
				init();
	        }
			Size sizeInput = DEF_INPUT_SIZE;
			
			Mat matInputImg = aMatInput.clone();
					
			if(APPLY_IMG_PADDING)
			{
				Mat matPaddedImg = null;
				Mat matROI = null;
				try {
					
					int iMaxPixels = Math.max(aMatInput.width(), aMatInput.height());
					matPaddedImg = new Mat(new Size(iMaxPixels,iMaxPixels), aMatInput.type(), Scalar.all(0));
					matROI = matPaddedImg.submat(0,aMatInput.rows(),0,aMatInput.cols());
					aMatInput.copyTo(matROI);
					
					matInputImg = matPaddedImg.clone();
				}
				finally
				{
					if(matPaddedImg!=null)
						matPaddedImg.release();
					if(matROI!=null)
						matROI.release();
				}
			}
			
//System.out.println("## Loaded aMatInput="+aMatInput);
			// Convert from BGR to RGB
			if(SWAP_RB_CHANNEL)
			{
				Imgproc.cvtColor(matInputImg, matInputImg, Imgproc.COLOR_BGR2RGB);
			}
			
			Mat matDnnImg = Dnn.blobFromImage(matInputImg, 1, sizeInput, Scalar.all(0), true, false);
			NET_YOLOX.setInput(matDnnImg);
//System.out.println("## Dnn Input Image="+matDnnImg);

	        // Run inference
	        List<Mat> outputs = new ArrayList<>();
	        NET_YOLOX.forward(outputs, NET_YOLOX.getUnconnectedOutLayersNames());
	        
	        Mat matResult = outputs.get(0);
//System.out.println("@@@ Inference Output="+matResult);
			
			matResult = postProcess(matResult, sizeInput);
//System.out.println("@@@ postProcess Output="+matResult);

	        float fConfidenceThreshold = DEF_CONFIDENCE_THRESHOLD;
	        float fNMSThreshold = DEF_NMS_THRESHOLD;
	        
	        List<Rect2d> outputBoxes 		= new ArrayList<>();
	        List<Float> outputConfidences 	= new ArrayList<>();
	        List<Integer> outputClassIds 	= new ArrayList<>();
	        
	        decodePredictions(matResult, sizeInput, outputBoxes, outputConfidences, outputClassIds, fConfidenceThreshold);

//System.out.println("@@@   Detection="+outputBoxes.size());
	        
	        if(outputBoxes.size()>0)
	        {
				Mat matOutputImg = matInputImg.clone();
				OpenCvUtil.resize(matOutputImg, (int)sizeInput.width, (int)sizeInput.height, false);
				
		        int[] indices = applyNMS(outputBoxes, outputConfidences, fConfidenceThreshold, fNMSThreshold);

//System.out.println("@@@   applyNMS="+indices.length);
		        
		        // Draw bounding boxes
		        for (int idx : indices) {
		        	
		            Rect2d box = outputBoxes.get(idx);
		            int classId = outputClassIds.get(idx);
		            
		            String label = OBJ_CLASSESS.get(classId) + ": " + String.format("%.2f", outputConfidences.get(idx));
		            
//System.out.println("    "+idx+"="+label+" "+box.tl()+" "+box.br());		            
		            Imgproc.rectangle(matOutputImg, new Point(box.x, box.y), new Point(box.x + box.width, box.y + box.height), new Scalar(0, 255, 0), 2);
		            Imgproc.putText(matOutputImg, label, new Point(box.x, box.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
		        }
				mapResult.put(IMLDetectionPlugin._KEY_MAT_OUTPUT, matOutputImg);
				mapResult.put(IMLDetectionPlugin._KEY_TOTAL_DETECTION, indices.length);
				//
				mapResult.put(IMLDetectionPlugin._KEY_THRESHOLD_DETECTION, fConfidenceThreshold);
				mapResult.put(IMLDetectionPlugin._KEY_THRESHOLD_NMS, fNMSThreshold);
				//
	        }
	        
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mapResult;
	}
	
	private void init()
	{
		NET_YOLOX = Dnn.readNetFromONNX( getModelFileName());
		
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
	
	private static Mat postProcess(Mat matOutputDetections, Size sizeInput)
	{
		int detectionCount = 0;
		matOutputDetections = matOutputDetections.reshape(1, new int[] {8400, 85});

		int[] hSizes = new int[STRIDES.length];
		int[] wSizes = new int[STRIDES.length];
		
		for (int i=0; i<STRIDES.length; i++) {
			hSizes[i] = Math.floorDiv((int)sizeInput.height, STRIDES[i]);
			wSizes[i] = Math.floorDiv((int)sizeInput.width, STRIDES[i]);
			//
			detectionCount += hSizes[i]*wSizes[i];
		}

		if (detectionCount != matOutputDetections.size(0)) {
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

                    double value0 = (matOutputDetections.get(detectionIdx, 0)[0] + x) * stride;
                    matOutputDetections.put(detectionIdx, 0, value0);

                    double value1 = (matOutputDetections.get(detectionIdx, 1)[0] + y) * stride;
                    matOutputDetections.put(detectionIdx, 1, value1);

                    double value2 = Math.exp(matOutputDetections.get(detectionIdx, 2)[0]) * stride;
                    matOutputDetections.put(detectionIdx, 2, value2);

                    double value3 = Math.exp(matOutputDetections.get(detectionIdx, 3)[0]) * stride;
                    matOutputDetections.put(detectionIdx, 3, value3);
                    
                    detectionIdx++;
                }
            }
        }
        
        return matOutputDetections;
    }

	
	private static int[] applyNMS(List<Rect2d> aBoxesList, List<Float> aConfidencesList, float CONFIDENCE_THRESHOLD, float NMS_THRESHOLD)
	{
        MatOfInt indices = new MatOfInt();

        if(aBoxesList.size()>0)
        {
	        // Apply Non-Maximum Suppression
	        MatOfRect2d boxesMat = new MatOfRect2d();
	        boxesMat.fromList(aBoxesList);
	        
	        MatOfFloat confidencesMat = new MatOfFloat();
	        confidencesMat.fromList(aConfidencesList);
	        
	        Dnn.NMSBoxes(boxesMat, confidencesMat, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
        }
        return indices.toArray();

	}

	private static void decodePredictions(
	        Mat matResult, Size imageSize, 
	        List<Rect2d> boxes, List<Float> confidences, List<Integer> classIds,
	        float CONFIDENCE_THRESHOLD) {
	    
        for (int i = 0; i < matResult.rows(); i++) 
        {
            Mat row = matResult.row(i);
            Mat scores = row.colRange(5, matResult.cols());
            Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
            float confidence = (float) mm.maxVal;
            
            if (confidence >= CONFIDENCE_THRESHOLD) {
                int classId = (int) mm.maxLoc.x;
                float[] data = new float[4];
                row.colRange(0, 4).get(0, 0, data);

                double centerX = data[0];
                double centerY = data[1];
                double width   = data[2];
                double height  = data[3];
                
                double left = centerX - (width / 2);
                double top 	= centerY - (height / 2);
                
                long lLeft  	= (long) Math.floor((left<0? 0: left));
                long lTop  		= (long) Math.floor((top<0? 0: top));
                long lWidth 	= (long) Math.floor(width);
        		long lHeight 	= (long) Math.floor(height);

                classIds.add(classId);
                confidences.add(confidence);
                boxes.add(new Rect2d(lLeft, lTop, lWidth, lHeight));
            }
        }
	}
}