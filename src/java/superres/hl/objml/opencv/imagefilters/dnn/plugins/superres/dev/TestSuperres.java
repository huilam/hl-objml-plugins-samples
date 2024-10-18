package hl.objml.opencv.imagefilters.dnn.plugins.superres.dev;

import hl.objml.opencv.imagefilters.dnn.plugins.superres.Upscale;
import hl.objml2.plugin.test.BaseTester;

public class TestSuperres extends BaseTester{

	public static void main(String[] args)
	{
		new BaseTester().testDetector(new Upscale());
	}
	
}