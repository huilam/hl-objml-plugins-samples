import hl.objml.opencv.objdetection.dnn.plugins.higherhrnet.HigherHRNetPoseDetector;


public class TestHigherHRNetPoseDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/faces");
		//
		test.testDetector(new HigherHRNetPoseDetector());
	}
}