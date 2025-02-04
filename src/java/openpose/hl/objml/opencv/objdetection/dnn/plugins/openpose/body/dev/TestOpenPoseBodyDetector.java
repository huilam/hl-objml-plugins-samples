package hl.objml.opencv.objdetection.dnn.plugins.openpose.body.dev;

import hl.objml.opencv.objdetection.dnn.plugins.openpose.body.OpenPoseBodyDetector;
import hl.objml2.plugin.test.BaseTester;

public class TestOpenPoseBodyDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		test.testDetector(new OpenPoseBodyDetector());
	}
}