package hl.objml2.dev.plugins.sample;

import hl.objml2.plugin.test.BaseTester;
import hl.objml2.plugin.ObjDetDnnBasePlugin;

public class TestSampleDnnDetector extends BaseTester {

	public static void main(String[] args)
	{
		ObjDetDnnBasePlugin detector = new SampleObjDetDnnPlugin();
		detector.addObjClassOfInterest(new String[]{"person"});
		//detector.addObjClassOfInterest(new String[]{"suitcase", "backpack", "handbag"});
		//detector.addObjClassOfInterest(new String[]{"truck", "car", "bus"});
		
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		//
		test.testDetector(detector);
	}
}