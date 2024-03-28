import yaml

from typing import Any, Dict

def get_params_from_cfg(cfg: str) -> Dict[str, Any]:

    f = open(cfg)
    cfg_infos = yaml.load_all(f, Loader=yaml.FullLoader)
    f.close()
    
    
    if "model_config" in cfg_info:
        model_config = oepn(cfg_info["model_config"])
        model_params = yaml.load(model_config, Loader=yaml.FullLoader)
        model_config.close()
    
        if os.path.exists(cfg_info["output_path"]):
            subprocess.run(["rm", "-rf", cfg_info["output_path"]])
        os.makedirs(cfg_info["output_path"])

        