package hl.objml.opencv.imagefilters.dnn.plugins.superres;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.opencv.core.Mat;
import org.opencv.dnn.Net;
import org.opencv.dnn_superres.DnnSuperResImpl;

import hl.objml2.common.FrameDetectionMeta;
import hl.objml2.plugin.ObjDetBasePlugin;


public class Upscale extends ObjDetBasePlugin {
	
	private DnnSuperResImpl superres = null;
	private Pattern pattModelNameNScale = Pattern.compile("([A-Z][A-Z,a-z]+)_x([2,3,4])\\.pb");

	@Override
	public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		List<Mat> listOutput = new ArrayList<>();
		
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

 				if(sAlgoName==null || sAlgoName.trim().length()==0 ||
 					sScale==null || sScale.trim().length()==0)
 				{
 					System.err.println("Error parsing "+super._model_filename+" Algo:"+sAlgoName+"  Scale:"+sScale);
 				}
	    		superres.setModel(sAlgoName.toLowerCase(), Integer.parseInt(sScale));
	    	}
	    	
	    	superres.readModel(super._model_filename);
    	}
    	
    	superres.setPreferableBackend(getDnnBackend());
    	superres.setPreferableTarget(getDnnTarget());
    	
    	Mat matOutput = new Mat();
    	superres.upsample(aMatInput, matOutput);
    	
    	listOutput = new ArrayList<>();
    	listOutput.add(matOutput);
    	
    	return listOutput;
	}
	
	@Override
	public Map<String,Object> parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		Map<String, Object> mapResult = new HashMap<String, Object>();
		Mat matOutput = aInferenceOutputMat.get(0);
		if(matOutput!=null)
		{
			//
			mapResult.put(ObjDetBasePlugin._KEY_OUTPUT_FRAME_ANNOTATED_IMG, matOutput);
			//
			FrameDetectionMeta meta = new FrameDetectionMeta();
			meta.setConfidence_threshold(getConfidenceThreshold());
			meta.setNms_threshold(DEF_NMS_THRESHOLD);
			meta.setObjml_model_name(getModelFileName());
			meta.setObjml_plugin_name(getPluginName());
			mapResult.put(ObjDetBasePlugin._KEY_OUTPUT_FRAME_DETECTION_META, meta);
			//
		}
		
		return mapResult;
	}

}