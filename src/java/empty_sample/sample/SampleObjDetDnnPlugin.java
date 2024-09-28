package sample;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;

import hl.objml2.plugin.ObjDetDnnBasePlugin;


public class SampleObjDetDnnPlugin extends ObjDetDnnBasePlugin {
	
	@Override
	public List<Mat> doInference(Mat aMatInput, JSONObject aCustomThresholdJson)
	{
		System.out.println("doInference()");
		Mat matDnnInput = Dnn.blobFromImage(
				aMatInput, 1.0/255.0, DEF_INPUT_SIZE, Scalar.all(0), true, false);
		NET_DNN.setInput(matDnnInput);
		List<Mat> listOutput = new ArrayList<>();
		NET_DNN.forward(listOutput, NET_DNN.getUnconnectedOutLayersNames());
		return listOutput;
	}
	
	@Override
	public Map<String,Object> parseDetections(
			List<Mat> aInferenceOutputMat, 
			Mat aMatInput, JSONObject aCustomThresholdJson)
	{
		System.out.println("parseDetections()");
		Map<String, Object> mapResult = new HashMap<String, Object>();
		for(int i=0; i<aInferenceOutputMat.size(); i++)
		{
			System.out.println(i+" = "+aInferenceOutputMat.get(i));
		}
		return mapResult;
	}
	
}