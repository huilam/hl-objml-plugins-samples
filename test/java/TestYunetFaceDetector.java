import hl.objml.opencv.objdetection.dnn.plugins.yunet.face.YunetFaceDetector;

public class TestYunetFaceDetector extends BaseTester {

	public static void main(String[] args)
	{
		testDetector(new YunetFaceDetector());
	}
}