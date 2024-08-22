import hl.objml.opencv.objdetection.MLDetectionBasePlugin;
import hl.objml.opencv.objdetection.dnn.plugins.yolo.YoloXDetector;

public class TestYoloXDetector extends BaseTester {

	public static void main(String[] args)
	{
		MLDetectionBasePlugin detector = new YoloXDetector();
		detector.addObjClassOfInterest("person");
		detector.addObjClassOfInterest("car");
		detector.addObjClassOfInterest("truck");
		detector.addObjClassOfInterest("bus");
		testDetector(detector);
	}
}