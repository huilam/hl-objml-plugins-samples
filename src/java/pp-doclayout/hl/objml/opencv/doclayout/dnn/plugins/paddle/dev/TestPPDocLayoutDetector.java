package hl.objml.opencv.doclayout.dnn.plugins.paddle.dev;

import hl.objml.opencv.doclayout.dnn.plugins.paddle.PPDocLayoutDetector;
import hl.objml2.plugin.test.BaseTester;

public class TestPPDocLayoutDetector extends hl.objml2.plugin.test.BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		test.testDetector(new PPDocLayoutDetector());
	}
}