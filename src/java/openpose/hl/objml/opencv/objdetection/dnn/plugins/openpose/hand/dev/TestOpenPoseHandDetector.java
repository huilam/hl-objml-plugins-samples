package hl.objml.opencv.objdetection.dnn.plugins.openpose.hand.dev;

import hl.objml.opencv.objdetection.dnn.plugins.openpose.hand.OpenPoseHandDetector;
import hl.objml2.plugin.test.BaseTester;

public class TestOpenPoseHandDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		test.testDetector(new OpenPoseHandDetector());
	}
}