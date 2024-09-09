import hl.objml.opencv.objdetection.dnn.plugins.detectron.Detectron2Detector;

public class TestDetectron2Detector extends BaseTester {

	public static void main(String[] args)
	{
		testDetector(new Detectron2Detector());
	}
}