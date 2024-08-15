import java.io.File;
import java.util.Map;
import java.util.Properties;

import org.opencv.core.Mat;

import hl.common.FileUtil;
import hl.objml.opencv.objdetection.MLDetectionBasePlugin;
import hl.opencv.util.OpenCvUtil;
import hl.plugin.image.IMLDetectionPlugin;

public class BaseTester {
	
	protected static File[] getTestImageFiles()
	{
		File folderImages = new File("./test/images/");
		
		if(folderImages.isDirectory())
		{
			return FileUtil.getFilesWithExtensions(folderImages, 
					new String[]{
							".jpg",
							".png"});
		}
		else
		{
			return null;
		}
		
	}
	
	protected static String saveImage(
			String aPluginName,
			Mat aMatImage, File aOutputFolder, String aOrigImgFileName)
	{
		if(!aOutputFolder.exists()) 
			aOutputFolder.mkdirs();
		
		String sOutputFileName = aPluginName+"_"+aOrigImgFileName;
		
		sOutputFileName += ".png";
	
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
		
		if(aDetector.isPluginOK())
		{
			Properties prop = aDetector.getPluginProps();
			
			System.out.println("Detector : "+aDetector.getPluginName()+" ("+aDetector.getPluginMLModelFileName()+")");
			
			
			int iKeyPrefix = "objml.mlmodel.".length();
			for(Object oKey : prop.keySet())
			{
				String sVal = prop.getProperty(oKey.toString());
				if(sVal!=null && sVal.trim().length()>0)
				{
					sVal = sVal.replace("\n", " ");
					
					if(sVal.length()>60) sVal = sVal.substring(0, 60)+" ... (truncated)";
					
					String sKey = oKey.toString().substring(iKeyPrefix);
					System.out.println("  - "+sKey+" : "+sVal);
				}
			}
			
			File fileFolder = new File("./test/images/output/"+System.currentTimeMillis());
			
			int i = 1;
			
			for(File fImg : getTestImageFiles())
			{
				System.out.println();
				System.out.print(" "+(i++)+". Perform test on "+fImg.getName()+" ... ");
				
				Mat matImg = MLDetectionBasePlugin.getCvMatFromFile(fImg);
				
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
	
}