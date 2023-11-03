import java.io.File;
import hl.objml.opencv.objdetection.IImgDetectorPlugin;
import hl.objml.opencv.objdetection.dnn.plugins.yunet.face.FaceDetector;


public class TestFaceDetector {

	
	public static void main(String[] args)
	{
		IImgDetectorPlugin detector = new FaceDetector();
		detector.detectImage(new File("./test/images/world-largest-selfie.jpg"));
	}
	
}