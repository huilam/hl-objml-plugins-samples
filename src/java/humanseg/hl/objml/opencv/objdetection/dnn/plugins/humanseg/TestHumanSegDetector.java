package hl.objml.opencv.objdetection.dnn.plugins.humanseg;

import hl.objml2.plugin.test.BaseTester;

public class TestHumanSegDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/internet");
		//
		test.testDetector(new HumanSegDetector());
	}
}