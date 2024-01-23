import java.util.HashMap;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Mat;

import hl.objml.opencv.objdetection.MLDetectionBasePlugin;
import hl.plugin.image.IMLDetectionPlugin;

public class EmptyDetectorPluginSample extends MLDetectionBasePlugin implements IMLDetectionPlugin {

	@Override
	public boolean isPluginOK() {
		return super.isPluginOK(getClass());
	}

	@Override
	public Map<String, Object> detect(Mat aImageFile, JSONObject aCustomThresholdJson) {
		// TODO Auto-generated method stub
		return new HashMap<String, Object>();
	}
	
}