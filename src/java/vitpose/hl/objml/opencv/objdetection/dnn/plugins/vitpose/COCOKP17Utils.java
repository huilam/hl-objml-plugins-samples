package hl.objml.opencv.objdetection.dnn.plugins.vitpose;

public class COCOKP17Utils {
	
    	public static String[] keypointsMapping = {
    	    "Nose",          // 0
    	    "Left Eye",      // 1
    	    "Right Eye",     // 2
    	    "Left Ear",      // 3
    	    "Right Ear",     // 4
    	    
    	    "Left Shoulder", // 5
    	    "Right Shoulder",// 6
    	    "Left Elbow",    // 7
    	    "Right Elbow",   // 8
    	    "Left Wrist",    // 9
    	    "Right Wrist",   // 10
    	    
    	    "Left Hip",      // 11
    	    "Right Hip",     // 12
    	    "Left Knee",     // 13
    	    "Right Knee",    // 14
    	    "Left Ankle",    // 15
    	    "Right Ankle"    // 16
    	};
    	
    	protected static int[] COCO_KP17_HEAD = {
    			 3 //Left Ear
    			,1 //Left Eye
    			,0 //Nose
    			,2 //Right Eye
    			,4 //Right Ear
    	};
    	
    	protected static int[] COCO_KP17_UPPER_BODY = {
     		 9  //Left Wrist
     	    ,7  //Left Elbow
   			,5  //Left Shoulder
   			,6  //Right Shoulder
   			,8  //Right Elbow
   			,10 //Right Wrist
    	};
    	
    	protected static int[] COCO_KP17_LOWER_BODY = {
        		16  //Left Ankle
        	   ,13  //Left Knee
      		   ,11  //Left Hip
      		   ,12  //Right Hip
        	   ,14  //Right Knee
       		   ,16  //Right Ankle
    	};
    	    	
}