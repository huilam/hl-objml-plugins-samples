package sample;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Mat;

import hl.objml2.plugin.ObjDetDnnBasePlugin;


public class SampleObjDetDnnPlugin extends ObjDetDnnBasePlugin {
	
	@Override
	public List<Mat> doInference(Mat aMatInput, JSONObject aCustomThresholdJson)
	{
		List<Mat> listOutput = new ArrayList<>();
		return listOutput;
	}
	
	@Override
	public Map<String,Object> parseDetections(
			List<Mat> aInferenceOutputMat, 
			Mat aMatInput, JSONObject aCustomThresholdJson)
	{
		Map<String, Object> mapResult = new HashMap<String, Object>();
		return mapResult;
	}
	
}