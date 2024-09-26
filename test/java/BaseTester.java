import java.io.File;
import java.util.Map;
import java.util.Properties;

import org.json.JSONObject;
import org.opencv.core.Mat;

import hl.common.FileUtil;
import hl.objml2.common.DetectedObj;
import hl.objml2.common.FrameDetectedObj;
import hl.objml2.plugin.ObjDetectionBasePlugin;
import hl.opencv.util.OpenCvUtil;

public class BaseTester {
	
	private String DEF_FOLDER_IMAGE = "./test/images/";
	
	private String FOLDER_IMAGE = DEF_FOLDER_IMAGE;
	
	protected void setTestImageFolder(String aImageFolder)
	{
		this.FOLDER_IMAGE = aImageFolder;
	}
	
	protected File[] getTestImageFiles()
	{
		File folderImages = new File(FOLDER_IMAGE);
		
		if(folderImages.isDirectory())
		{
			return FileUtil.getFilesWithExtensions(folderImages, 
					new String[]{
							".jpg",
							".png"});
		}
		else
		{
			return new File[] {};
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
	
	public void testDetector(ObjDetectionBasePlugin aDetector)
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
			System.out.println("Detector : "+aDetector.getPluginName()+" ("+aDetector.getPluginMLModelFileName()+")");
			System.out.println("isPluginOK : "+aDetector.isPluginOK());

			Properties prop = aDetector.getPluginProps();
			
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
				
				Mat matImg = ObjDetectionBasePlugin.getCvMatFromFile(fImg);
				
				long lInferenceStart = System.currentTimeMillis();
				Map<String, Object> mapResult = aDetector.detect(matImg, null);
				long lInferenceEnd = System.currentTimeMillis();
				
				if(mapResult!=null)
				{
					long lInferenceMs =  lInferenceEnd-lInferenceStart;
					
					Integer outputTotalDetections = (Integer) mapResult.get(ObjDetectionBasePlugin._KEY_OUTPUT_TOTAL_COUNT);
					
					
					System.out.println();
					System.out.println("     - Inference Model File : "+new File(aDetector.getPluginMLModelFileName()).getName());
					System.out.println("     - Inference Input Size : "+matImg.size().toString());
					System.out.println("     - Inference Time (Ms)  : "+lInferenceMs);
					
					JSONObject jsonDetection = (JSONObject) mapResult.get(ObjDetectionBasePlugin._KEY_OUTPUT_DETECTION_JSON);
					if(jsonDetection!=null)
					{
						FrameDetectedObj objs = new FrameDetectedObj();
						objs.fromJson(jsonDetection);
						//
						System.out.println("     - ObjClass Names : "+String.join(",", objs.getObjClassNames()));
						System.out.println("     - Total Detection : "+(outputTotalDetections==null?"(missing data)":outputTotalDetections));
					}
					else
					{
						System.out.println("     - Detection JSON : "+jsonDetection);
					}
					
					Mat matOutput = (Mat) mapResult.get(ObjDetectionBasePlugin._KEY_OUTPUT_ANNOTATED_MAT);
					
					if(matOutput!=null && !matOutput.empty() && outputTotalDetections>0)
					{
						String savedFileName = 
								saveImage(aDetector.getPluginName(), 
								matOutput, 
								fileFolder, fImg.getName());
						
						if(savedFileName!=null)
							System.out.println("     - [saved] "+savedFileName);
					}
				}
				else
				{
					System.out.println("     - No result found.");
				}
			}
			
		}		
	}
	
}