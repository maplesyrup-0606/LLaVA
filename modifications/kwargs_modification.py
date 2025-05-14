from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

class PatchedLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):

    def _validate_model_kwargs(self, model_kwargs) :
        model_kwargs.pop("image_infos", None)
        return super()._validate_model_kwargs(model_kwargs)
    