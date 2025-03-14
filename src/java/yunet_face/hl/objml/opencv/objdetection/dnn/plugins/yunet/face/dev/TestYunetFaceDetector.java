package hl.objml.opencv.objdetection.dnn.plugins.yunet.face.dev;

import hl.objml.opencv.objdetection.dnn.plugins.yunet.face.YunetFaceDetector;
import hl.objml2.plugin.test.BaseTester;

public class TestYunetFaceDetector extends BaseTester {

	public static void main(String[] args)
	{
		new BaseTester().testDetector(new YunetFaceDetector());
	}
}