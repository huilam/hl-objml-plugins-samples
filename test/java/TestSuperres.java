import hl.objml.opencv.imagefilters.dnn.plugins.superres.Upscale;

public class TestSuperres extends BaseTester{

	public static void main(String[] args)
	{
		testDetector(new Upscale());
	}
	
}