package hl.objml.opencv.objdetection.dnn.plugins.vitpose;

import hl.objml2.dev.plugins.test.BaseTester;

public class TestVitPoseDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		test.testDetector(new VitPoseDetector());
	}
}