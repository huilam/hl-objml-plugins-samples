package hl.objml.opencv.objdetection.dnn.plugins.mediapipe.face;

import java.util.List;
import java.util.Map;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;

import hl.objml.opencv.objdetection.dnn.plugins.mediapipe.BaseMediaPipeDetector;
import hl.objml2.common.DetectedObj;

public class MediaPipeFaceDetector extends BaseMediaPipeDetector {
	
	/**
	 *  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
	 */ 
	
    @Override
    protected void decodePredictions(
            final List<Mat> listMatOutputs, 
            final Mat aMatInput,
            List<DetectedObj> aDetectedObj,
            final double aConfidenceThreshold) {

        Mat o1 = listMatOutputs.get(0);  // 1×2016×1 = Confidence scores
        Mat o2 = listMatOutputs.get(1);  // 1×2016×18 = Bounding boxes & keypoints 

        float imgW = aMatInput.width();
        float imgH = aMatInput.height();
        
        float modelW = (float) getImageInputSize().width;
        float modelH = (float) getImageInputSize().height;
        
        float scaleX = modelW / imgW;
        float scaleY = modelH / imgH;
        
        int iTotalDetections = o1.size(1);
        int iDataSize = 16;
        
        
        Mat output1 = o1.reshape(1, new int[]{iTotalDetections, o1.size(2)});
        Mat output2 = o2.reshape(1, new int[]{iTotalDetections, iDataSize});
        
        Map<Point, Float> mapDetections = 
        		getTopDetections(1, output1, aConfidenceThreshold);
        
        for(Object oIdx : mapDetections.keySet())
        {
        	Point pt = (Point)oIdx;
        	float confidence = (float) mapDetections.get(pt);
    
            float[] bestDetection = new float[iDataSize];
            output2.get((int)pt.y, (int)pt.x, bestDetection);

            // Extract bounding box values
            float cx = bestDetection[0];
            float cy = bestDetection[1];
            float width  = bestDetection[2];
            float height = bestDetection[3];

            // Convert from center-based to top-left based
            float x = Math.abs(cx - (width/2));
            float y = Math.abs(cy - (height/2));

            // Clip bounding box within image bounds
            /**
            x = Math.max(0, Math.min(x, imgW - 1));
            y = Math.max(0, Math.min(y, imgH - 1));
            width = Math.min(width, imgW - x);
            height = Math.min(height, imgH - y);
            **/

            System.out.println("\n>> " + pt + " confidence=" + confidence + " (x:"+x+",y:"+y+",w:"+width+",h:"+height+")");
            Rect2d box = new Rect2d(
            		Math.round(x / scaleX), 
            		Math.round(y / scaleY), 
            		Math.round(width / scaleX), 
            		Math.round(height / scaleY));
 
            DetectedObj obj = new DetectedObj(0, "Face", box, confidence);
            aDetectedObj.add(obj);
        }

        ANNOTATE_OUTPUT_IMG = true;
    }
}