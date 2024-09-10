import hl.objml.opencv.objdetection.dnn.plugins.dbtext.DBTextDetector;

public class TestDBTextDetector extends BaseTester {

	public static void main(String[] args)
	{
		new BaseTester().testDetector(new DBTextDetector());
	}
}