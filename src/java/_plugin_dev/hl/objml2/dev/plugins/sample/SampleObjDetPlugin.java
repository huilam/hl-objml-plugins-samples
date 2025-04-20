package hl.objml2.dev.plugins.sample;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.dnn.Net;

import hl.objml2.plugin.MLPluginConfigProp;
import hl.objml2.plugin.MLPluginFrameOutput;
import hl.objml2.plugin.ObjDetBasePlugin;


public class SampleObjDetPlugin extends ObjDetBasePlugin {
	
	@Override
    public List<Mat> doInference(Mat aMatInput, Net aDnnNet)
	{
		List<Mat> listOutput = new ArrayList<>();
		return listOutput;
	}
	
	@Override
    public MLPluginFrameOutput parseDetections(Mat aMatInput, List<Mat> aInferenceOutputMat)
	{
		MLPluginFrameOutput frameOutput = null;
		return frameOutput;
	}
	
	@Override
	public MLPluginConfigProp prePropInit(MLPluginConfigProp aProps) 
	{
		return aProps;
	}
	
}