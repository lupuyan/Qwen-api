from utils import torch_gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

class Client:
    model = None
    tokenizer = None
    def init_cfg(self,):
        model_id = 'Qwen/Qwen-7B-Chat'
        # 请注意：我们的分词器做了对特殊token攻击的特殊处理。因此，你不能输入诸如<|endoftext|>这样的token，会出现报错。
        # 如需移除此策略，你可以加入这个参数`allowed_special`，可以接收"all"这个字符串或者一个特殊tokens的`set`。
        # 举例: tokens = tokenizer(text, allowed_special="all")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # 建议先判断当前机器是否支持BF16，命令如下所示：
        # import torch
        # torch.cuda.is_bf16_supported()
        # 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
        # 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        # 使用CPU进行推理，需要约32GB内存
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
        # 默认使用fp32精度
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                     trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_id,
                                                                   trust_remote_code=True)  # 可指定不同的生成长度、top_p等相关超参

    def stream_chat(self, query, chat_history=[], streaming: bool = True):

        for result in self.model.chat(self.tokenizer, query, history=chat_history, stream=streaming):
            torch_gc()
            # history[-1][0] = query
            response = {"query": query,
                        "result": result}
            yield response
            torch_gc()
