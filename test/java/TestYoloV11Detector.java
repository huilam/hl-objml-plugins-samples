import hl.objml.opencv.objdetection.dnn.plugins.yolov11.YoloV11Detector;
import hl.objml2.plugin.ObjDetDnnBasePlugin;

public class TestYoloV11Detector extends BaseTester {

	public static void main(String[] args)
	{
		ObjDetDnnBasePlugin detector = new YoloV11Detector();
		detector.addObjClassOfInterest(new String[]{"person"});
		detector.addObjClassOfInterest(new String[]{"handbag", "suitcase", "backpack"});
		//detector.addObjClassOfInterest(new String[]{"bottle"});
		//detector.addObjClassOfInterest(new String[]{"truck", "car", "bus"});
		BaseTester test = new BaseTester();
		test.setTestImageFolder("./test/images/coco");
		test.testDetector(detector);
	}
}