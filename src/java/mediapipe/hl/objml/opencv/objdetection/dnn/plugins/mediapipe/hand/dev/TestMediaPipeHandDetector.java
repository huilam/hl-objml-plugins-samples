package hl.objml.opencv.objdetection.dnn.plugins.mediapipe.hand.dev;

import hl.objml.opencv.objdetection.dnn.plugins.mediapipe.hand.MediaPipeHandDetector;
import hl.objml2.plugin.test.BaseTester;

public class TestMediaPipeHandDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/internet/hands");
		test.testDetector(new MediaPipeHandDetector());
	}
}