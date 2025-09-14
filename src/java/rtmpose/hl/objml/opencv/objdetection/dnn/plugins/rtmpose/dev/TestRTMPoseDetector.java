package hl.objml.opencv.objdetection.dnn.plugins.rtmpose.dev;

import hl.objml2.plugin.test.BaseTester;
import hl.objml.opencv.objdetection.dnn.plugins.rtmpose.RTMPoseDetector;
import hl.objml2.plugin.ObjDetDnnBasePlugin;

public class TestRTMPoseDetector extends BaseTester {

	public static void main(String[] args)
	{
		ObjDetDnnBasePlugin detector = new RTMPoseDetector();
		detector.addObjClassOfInterest(new String[]{"person"});
		//detector.addObjClassOfInterest(new String[]{"suitcase", "backpack", "handbag"});
		//detector.addObjClassOfInterest(new String[]{"truck", "car", "bus"});
		
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		//
		test.testDetector(detector);
	}
}