import java.io.File;

import hl.objml.opencv.objdetection.IImgDetectorPlugin;
import hl.objml.opencv.objdetection.dnn.plugins.superres.Upscale;

public class TestSuperres {

	public static void main(String[] args)
	{
		IImgDetectorPlugin upscale = new Upscale();
		upscale.detectImage(new File("./test/images/world-largest-selfie.jpg"));

	}
	
}