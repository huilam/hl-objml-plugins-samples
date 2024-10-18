package hl.objml.opencv.objdetection.dnn.plugins.yolov11.dev;

import hl.objml.opencv.objdetection.dnn.plugins.yolov11.YoloV11Detector;
import hl.objml2.plugin.test.BaseTester;

public class TestYolov11Detector extends hl.objml2.plugin.test.BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		test.testDetector(new YoloV11Detector());
	}
}