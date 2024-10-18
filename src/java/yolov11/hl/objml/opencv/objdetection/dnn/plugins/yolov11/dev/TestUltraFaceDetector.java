package hl.objml.opencv.objdetection.dnn.plugins.yolov11.dev;

import hl.objml.opencv.objdetection.dnn.plugins.ultraface.UltraFaceDetector;
import hl.objml2.plugin.test.BaseTester;

public class TestUltraFaceDetector extends hl.objml2.plugin.test.BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/temp");
		test.testDetector(new UltraFaceDetector());
	}
}