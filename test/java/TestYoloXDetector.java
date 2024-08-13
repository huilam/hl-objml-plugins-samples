import hl.objml.opencv.objdetection.dnn.plugins.yolo.YoloXDetector;

public class TestYoloXDetector extends BaseTester {

	public static void main(String[] args)
	{
		testDetector(new YoloXDetector());
	}
}