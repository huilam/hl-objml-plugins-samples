import sample.SampleObjDetDnnPlugin;


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