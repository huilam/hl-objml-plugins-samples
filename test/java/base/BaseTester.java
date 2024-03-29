package base;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.opencv.core.Mat;

import hl.objml.opencv.objdetection.MLDetectionBasePlugin;
import hl.opencv.util.OpenCvUtil;
import hl.plugin.image.IMLDetectionPlugin;

public class BaseTester {
	
	protected static List<File> getTestImageFiles()
	{
		List<File> listImage = new ArrayList<File>();
		listImage.add(new File("./test/images/world-largest-selfie.jpg"));
		
		listImage.add(new File("./test/images/dashcam_streetview.jpg"));
		
		return listImage;
	}
	
	protected static String saveImage(
			String aPluginName,
			Mat aMatImage, File aOutputFolder, String aOrigImgFileName)
	{
		if(!aOutputFolder.exists()) 
			aOutputFolder.mkdirs();
		
		String sOutputFileName = aPluginName+"_"+aOrigImgFileName;
	
		boolean isSaved = OpenCvUtil.saveImageAsFile(aMatImage, aOutputFolder.getAbsolutePath()+"/"+sOutputFileName);
		
		if(isSaved)
			return aOutputFolder.getName()+"/"+sOutputFileName;
		return null;
	}
	
	public static void testDetector(IMLDetectionPlugin aDetector)
	{
		OpenCvUtil.initOpenCV();
		
		try {
			Thread.sleep(200);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		aDetector.isPluginOK();
		
		File fileFolder = new File("./test/images/output/"+System.currentTimeMillis());
		
		int i = 1;
		
		for(File fImg : getTestImageFiles())
		{
			System.out.println();
			System.out.print(" "+(i++)+". Perform test on "+fImg.getName()+" ... ");
			
			Mat matImg = MLDetectionBasePlugin.getCvMatFromFile(fImg);
			OpenCvUtil.removeAlphaChannel(matImg);
			
			Map<String, Object> mapResult = aDetector.detect(matImg, null);
			
			System.out.println("     - Result : "+(mapResult!=null?mapResult.size():0));
			
			if(mapResult!=null)
			{
				Mat matOutput = (Mat) mapResult.get(IMLDetectionPlugin._KEY_MAT_OUTPUT);
				
				if(matOutput!=null && !matOutput.empty())
				{
					String savedFileName = 
							saveImage(aDetector.getPluginName(), 
							matOutput, 
							fileFolder, fImg.getName());
					
					if(savedFileName!=null)
						System.out.println("     - [saved] "+savedFileName);
				}
				else
				{
					int idx = 0;
					for(String key : mapResult.keySet())
					{
						System.out.println("    ["+idx+"] "+key+" - "+mapResult.get(key));
						idx++;
					}
				}
			}
			
		}		
	}
	
}