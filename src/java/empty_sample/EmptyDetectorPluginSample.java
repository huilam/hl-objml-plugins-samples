import java.util.HashMap;
import java.util.Map;

import org.json.JSONObject;
import org.opencv.core.Mat;

import hl.objml2.plugin.IObjDetectionPlugin;
import hl.objml2.plugin.ObjDetectionBasePlugin;

public class EmptyDetectorPluginSample extends ObjDetectionBasePlugin implements IObjDetectionPlugin {

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