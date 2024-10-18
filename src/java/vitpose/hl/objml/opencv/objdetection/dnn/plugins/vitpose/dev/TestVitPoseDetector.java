package hl.objml.opencv.objdetection.dnn.plugins.vitpose.dev;

import hl.objml.opencv.objdetection.dnn.plugins.vitpose.VitPoseDetector;
import hl.objml2.plugin.test.BaseTester;

public class TestVitPoseDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		test.testDetector(new VitPoseDetector());
	}
}