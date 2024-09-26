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
    
    	public static int[][] keypointConnections = {
    	    {5, 6},   // Left Shoulder to Right Shoulder
    	    {5, 7},   // Left Shoulder to Left Elbow
    	    {7, 9},   // Left Elbow to Left Wrist
    	    {6, 8},   // Right Shoulder to Right Elbow
    	    {8, 10},  // Right Elbow to Right Wrist
    	    {5, 11},  // Left Shoulder to Left Hip
    	    {6, 12},  // Right Shoulder to Right Hip
    	    {11, 12}, // Left Hip to Right Hip
    	    {11, 13}, // Left Hip to Left Knee
    	    {13, 15}, // Left Knee to Left Ankle
    	    {12, 14}, // Right Hip to Right Knee
    	    {14, 16}, // Right Knee to Right Ankle
    	    {0, 1},   // Nose to Left Eye
    	    {0, 2},   // Nose to Right Eye
    	    {1, 3},   // Left Eye to Left Ear
    	    {2, 4}    // Right Eye to Right Ear
    	};
}