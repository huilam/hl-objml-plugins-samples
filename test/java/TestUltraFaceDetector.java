import hl.objml.opencv.objdetection.dnn.plugins.ultraface.UltraFaceDetector;

public class TestUltraFaceDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		test.testDetector(new UltraFaceDetector());
	}
}