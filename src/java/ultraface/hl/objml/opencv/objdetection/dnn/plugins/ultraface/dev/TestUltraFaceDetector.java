package hl.objml.opencv.objdetection.dnn.plugins.ultraface.dev;

import hl.objml.opencv.objdetection.dnn.plugins.ultraface.UltraFaceDetector;
import hl.objml2.plugin.test.BaseTester;

public class TestUltraFaceDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/internet/person");
		test.testDetector(new UltraFaceDetector());
	}
}