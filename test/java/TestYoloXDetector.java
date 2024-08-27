import hl.objml.opencv.objdetection.ObjDetectionBasePlugin;
import hl.objml.opencv.objdetection.dnn.plugins.yolo.YoloXDetector;

public class TestYoloXDetector extends BaseTester {

	public static void main(String[] args)
	{
		ObjDetectionBasePlugin detector = new YoloXDetector();
		detector.addObjClassOfInterest(new String[]{"person"});
		detector.addObjClassOfInterest(new String[]{"truck", "car", "bus"});
		testDetector(detector);
	}
}