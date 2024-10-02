import hl.objml.opencv.objdetection.dnn.plugins.vitpose.VitPoseDetector;

public class TestVitPoseDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/faces");
		test.testDetector(new VitPoseDetector());
	}
}