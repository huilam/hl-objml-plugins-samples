package hl.objml2.dev.plugins.sample;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import hl.objml2.plugin.MLPluginFrameOutput;
import hl.objml2.plugin.ObjDetDnnBasePlugin;


public class SampleObjDetDnnPlugin extends ObjDetDnnBasePlugin {
	
	@Override
    public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		System.out.println("doInference()");
		Mat matDnnInput = Dnn.blobFromImage(
				aMatInput, 1.0/255.0, getImageInputSize(), Scalar.all(0), true, false);
		aDnnNet.setInput(matDnnInput);
		List<Mat> listOutput = new ArrayList<>();
		aDnnNet.forward(listOutput, aDnnNet.getUnconnectedOutLayersNames());
		return listOutput;
	}
	
	@Override
    public MLPluginFrameOutput parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		MLPluginFrameOutput frameOutput = null;
		System.out.println("parseDetections()");
		for(int i=0; i<aInferenceOutputMat.size(); i++)
		{
			System.out.println(i+" = "+aInferenceOutputMat.get(i));
		}
		return frameOutput;
	}
	
	@Override
	public Properties prePropInit(Properties aProps) 
	{
		return aProps;
	}

	
}