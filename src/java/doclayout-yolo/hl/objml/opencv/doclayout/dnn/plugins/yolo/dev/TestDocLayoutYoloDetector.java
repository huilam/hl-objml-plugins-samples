package hl.objml.opencv.doclayout.dnn.plugins.yolo.dev;

import hl.objml.opencv.doclayout.dnn.plugins.yolo.DocLayoutYolo10Detector;
import hl.objml2.plugin.test.BaseTester;

public class TestDocLayoutYoloDetector extends hl.objml2.plugin.test.BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/pdf");
		test.testDetector(new DocLayoutYolo10Detector());
	}
}