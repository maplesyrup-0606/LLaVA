from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from typing import Any, Dict

class PatchedLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):

    def _validate_model_kwargs(self, model_kwargs : Dict[str, Any]) :
        image_infos = model_kwargs.pop("image_infos", None)
        super()._validate_model_kwargs(model_kwargs)
        
        if image_infos is not None :
            model_kwargs['image_infos'] = image_infos
        
        return
    