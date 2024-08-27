package hl.objml.opencv.imagefilters.dnn.plugins.superres;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.json.JSONObject;
import org.opencv.core.Mat;
import org.opencv.dnn_superres.DnnSuperResImpl;

import hl.objml.opencv.objdetection.ObjDetectionBasePlugin;


public class Upscale extends ObjDetectionBasePlugin {
	
	private DnnSuperResImpl superres = null;
	private Pattern pattModelNameNScale = Pattern.compile("([A-Z][A-Z,a-z]+)_x([2,3,4])\\.pb");
	
	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}
	
	@Override
	public Map<String, Object> detect(Mat aMatInput, JSONObject aCustomThresholdJson) {
		Map<String, Object> mapResult = new HashMap<String, Object>();
		try {
			Mat matOutput = upScaling(aMatInput);
			if(matOutput!=null)
			{
				mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_ANNOTATED_MAT, matOutput);
				mapResult.put(ObjDetectionBasePlugin._KEY_OUTPUT_TOTAL_COUNT, 1);
				//
				mapResult.put(ObjDetectionBasePlugin._KEY_THRESHOLD_DETECTION, -1);
				mapResult.put(ObjDetectionBasePlugin._KEY_THRESHOLD_NMS, -1);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mapResult;
	}
	
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
}