package sample;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.opencv.core.Mat;
import org.opencv.dnn.Net;

import hl.objml2.plugin.ObjDetBasePlugin;


public class SampleObjDetPlugin extends ObjDetBasePlugin {
	
	@Override
    public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		List<Mat> listOutput = new ArrayList<>();
		return listOutput;
	}
	
	@Override
    public Map<String,Object> parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		Map<String, Object> mapResult = new HashMap<String, Object>();
		return mapResult;
	}
	
	@Override
	public Properties prePropInit(Properties aProps) 
	{
		return aProps;
	}
	
}