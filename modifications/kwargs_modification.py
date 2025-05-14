from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

class PatchedLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):
    # @classmethod
    # def from_pretrained(cls, *args, **kwargs) :
    #     model = super().from_pretrained(*args, **kwargs)

    #     print(model.config)

    #     from modifications.attention_modification import PatchedLlamaAttention

    #     for layer in model.model.layers :
    #         old_attn = layer.self_attn 
    #         new_attn = PatchedLlamaAttention(model.config, layer_idx=old_attn.layer_idx)
            
    #         new_attn.load_state_dict(old_attn.state_dict())

    #         new_attn = new_attn.to(device=old_attn.q_proj.weight.device, dtype=old_attn.q_proj.weight.dtype)
    #         layer.self_attn = new_attn
        
    #     return model


    def _validate_model_kwargs(self, model_kwargs) :
        model_kwargs.pop("image_infos", None)
        return super()._validate_model_kwargs(model_kwargs)