package hl.objml.opencv.objdetection.dnn.plugins.ultraface;

import hl.objml2.dev.plugins.test.BaseTester;

public class TestUltraFaceDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		test.testDetector(new UltraFaceDetector());
	}
}