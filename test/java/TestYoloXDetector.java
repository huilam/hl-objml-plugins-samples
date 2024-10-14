import hl.objml.opencv.objdetection.dnn.plugins.yolox.YoloXDetector;
import hl.objml2.dev.plugins.test.BaseTester;
import hl.objml2.plugin.ObjDetDnnBasePlugin;

public class TestYoloXDetector extends BaseTester {

	public static void main(String[] args)
	{
		ObjDetDnnBasePlugin detector = new YoloXDetector();
		detector.addObjClassOfInterest(new String[]{"person"});
		detector.addObjClassOfInterest(new String[]{"suitcase", "backpack", "handbag"});
		//detector.addObjClassOfInterest(new String[]{"truck", "car", "bus"});
		//
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/test01");
		test.testDetector(detector);
	}
}