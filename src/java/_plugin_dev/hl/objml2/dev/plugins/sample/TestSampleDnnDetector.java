package hl.objml2.dev.plugins.sample;

import hl.objml2.dev.plugins.test.BaseTester;

public class TestSampleDnnDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		//
		test.testDetector(new SampleObjDetDnnPlugin());
	}
}