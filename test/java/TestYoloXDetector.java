import hl.objml.opencv.objdetection.dnn.plugins.yolo.YoloXDetector;
import hl.objml2.plugin.ObjDetectionBasePlugin;

public class TestYoloXDetector extends BaseTester {

	public static void main(String[] args)
	{
		ObjDetectionBasePlugin detector = new YoloXDetector();
		detector.addObjClassOfInterest(new String[]{"person"});
		//detector.addObjClassOfInterest(new String[]{"truck", "car", "bus"});
		//
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/bikes-ii");
		test.testDetector(detector);
	}
}