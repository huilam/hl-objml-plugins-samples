package hl.objml.opencv.objdetection.dnn.plugins.text.detect.recog.dev;

import java.io.File;

import hl.objml.opencv.objdetection.dnn.plugins.text.detect.recog.DBTextRecognizer;
import hl.objml2.plugin.test.BaseTester;

public class TestDBTextRecognizer extends BaseTester {

	public static void main(String[] args)
	{
		BaseTester tester = new BaseTester();
		tester.setIsAutoSaveOutputMatAsFile(true);
		tester.setTestImageFolder(new File("./test/images/text").getAbsolutePath());
		tester.testDetector(new DBTextRecognizer());
	}
}