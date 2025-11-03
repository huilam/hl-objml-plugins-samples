package hl.objml.opencv.objdetection.dnn.plugins.text.detect.dev;

import java.io.File;

import hl.objml.opencv.objdetection.dnn.plugins.text.detect.DBTextDetector;
import hl.objml2.plugin.test.BaseTester;

public class TestDBTextDetector extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester tester = new BaseTester();
		tester.setIsAutoSaveOutputMatAsFile(true);
		tester.setTestImageFolder(new File("./test/images/text").getAbsolutePath());
		tester.testDetector(new DBTextDetector());
	}
}