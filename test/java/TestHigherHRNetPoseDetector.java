import hl.objml.opencv.objdetection.dnn.plugins.higherhrnet.HigherHRNetPoseDetector;
import hl.objml2.dev.plugins.test.BaseTester;


public class TestHigherHRNetPoseDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		//
		test.testDetector(new HigherHRNetPoseDetector());
	}
}