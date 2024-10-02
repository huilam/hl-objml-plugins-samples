package sample;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import hl.objml2.plugin.ObjDetDnnBasePlugin;


public class SampleObjDetDnnPlugin extends ObjDetDnnBasePlugin {
	
	@Override
    public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		System.out.println("doInference()");
		Mat matDnnInput = Dnn.blobFromImage(
				aMatInput, 1.0/255.0, DEF_INPUT_SIZE, Scalar.all(0), true, false);
		aDnnNet.setInput(matDnnInput);
		List<Mat> listOutput = new ArrayList<>();
		aDnnNet.forward(listOutput, aDnnNet.getUnconnectedOutLayersNames());
		return listOutput;
	}
	
	@Override
    public Map<String,Object> parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		System.out.println("parseDetections()");
		Map<String, Object> mapResult = new HashMap<String, Object>();
		for(int i=0; i<aInferenceOutputMat.size(); i++)
		{
			System.out.println(i+" = "+aInferenceOutputMat.get(i));
		}
		return mapResult;
	}
	
	@Override
	public Properties prePropInit(Properties aProps) 
	{
		return aProps;
	}

	
}