import base.BaseTester;
import hl.objml.opencv.objdetection.dnn.plugins.yunet.face.FaceDetector;

public class TestFaceDetector extends BaseTester {

	public static void main(String[] args)
	{
		testDetector(new FaceDetector());
	}
}