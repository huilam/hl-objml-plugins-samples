import hl.objml2.dev.plugins.sample.SampleObjDetDnnPlugin;
import hl.objml2.dev.plugins.test.BaseTester;


public class TestDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		//
		
		//
		test.testDetector(new SampleObjDetDnnPlugin());
	}
}