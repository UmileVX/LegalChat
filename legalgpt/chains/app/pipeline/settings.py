# from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline


class SetParams:
    def __init__(self, my_llm, **new_params):
        self.pipeline = my_llm.pipeline
        self._old_params = {**self.pipeline._forward_params}
        self._new_params = new_params

    def __enter__(self):
        self.pipeline._forward_params.update(**self._new_params)

    def __exit__(self ,type, value, traceback):
        for k in self._new_params.keys(): 
            del self.pipeline._forward_params[k]
        self.pipeline._forward_params.update(self._old_params)


def inference_hf_pipeline_with_params(
    pipeline: HuggingFacePipeline,
    input_text: str,
    prompt_text: str="<s>[INST]<<SYS>>Hello World!<</SYS>>{}[/INST]",
    max_new_tokens: int=2,
    eos_token_id: list=[2],
    # **kwargs
):
    prompt = prompt_text.format(input_text)
    with SetParams(pipeline, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id):
        response = pipeline.predict(prompt)
    return response
