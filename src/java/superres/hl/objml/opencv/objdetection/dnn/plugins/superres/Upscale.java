package hl.objml.opencv.objdetection.dnn.plugins.superres;

import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.opencv.core.Mat;
import org.opencv.dnn_superres.DnnSuperResImpl;

import hl.objml.opencv.objdetection.ImgMLBasePlugin;
import hl.plugin.image.IImgDetectorPlugin;


public class Upscale extends ImgMLBasePlugin implements IImgDetectorPlugin {
	
	private DnnSuperResImpl superres = null;
	
	private Pattern pattModelNameNScale = Pattern.compile("([A-Z][A-Z,a-z]+)_x([2,3,4])\\.pb");
	
    public Mat upScaling(Mat aMatInput) throws Exception
    {
    	if(!isPluginOK() || aMatInput==null)
    		return null;
    	
    	if(superres==null)
    	{
	    	superres = DnnSuperResImpl.create();
	    	
	    	Matcher m = pattModelNameNScale.matcher(super._model_filename);
	    	if(m.find())
	    	{
	    		String sAlgoName = m.group(1); //FSRCNN
	    		String sScale = m.group(2); //4

//System.out.println(_model_filename+" = "+sAlgoName.toLowerCase()+" x"+sScale);

 				if(sAlgoName==null || sAlgoName.trim().length()==0 ||
 					sScale==null || sScale.trim().length()==0)
 				{
 					System.err.println("Error parsing "+super._model_filename+" Algo:"+sAlgoName+"  Scale:"+sScale);
 				}
	    		superres.setModel(sAlgoName.toLowerCase(), Integer.parseInt(sScale));
	    	}
	    	
	    	superres.readModel(super._model_filename);
    	}
    	
    	Mat matOutput = new Mat();
    	superres.upsample(aMatInput, matOutput);
    	return matOutput;
    }

	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}

	@Override
	public Properties getPluginProps() {
		return super.getPluginProps();
	}
	
	@Override
	public String getPluginName() {
		return super.getModelName();
	}
	
	@Override
	public String getPluginMLModelFileName()
	{
		return super.getModelFileName();
	}
	
	@Override
	public Map<String, Object> detectImage(Mat aMatInput) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		try {
			Mat matOutput = upScaling(aMatInput);
			if(matOutput!=null)
			{
				mapResult.put(IImgDetectorPlugin._KEY_MAT_OUTPUT, matOutput);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mapResult;
	}
    
}