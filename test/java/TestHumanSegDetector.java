import hl.objml.opencv.objdetection.dnn.plugins.humanseg.HumanSegDetector;

public class TestHumanSegDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		//
		test.testDetector(new HumanSegDetector());
	}
}