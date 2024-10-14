package hl.objml.opencv.imagefilters.dnn.plugins.superres;

import hl.objml2.dev.plugins.test.BaseTester;

public class TestSuperres extends BaseTester{

	public static void main(String[] args)
	{
		new BaseTester().testDetector(new Upscale());
	}
	
}