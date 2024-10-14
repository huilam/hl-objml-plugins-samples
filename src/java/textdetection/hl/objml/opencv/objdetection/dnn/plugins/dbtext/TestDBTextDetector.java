package hl.objml.opencv.objdetection.dnn.plugins.dbtext;

import hl.objml2.dev.plugins.test.BaseTester;

public class TestDBTextDetector extends BaseTester {

	public static void main(String[] args)
	{
		new BaseTester().testDetector(new DBTextDetector());
	}
}