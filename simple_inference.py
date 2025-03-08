# refactored by ChatGPT-o1
# https://chatgpt.com/share/67cc5951-ba04-8009-8de7-a3448411bf63

import argparse
import os
import sys
import re
import time
from functools import partial

import torch
import torchaudio
import torchaudio.functional as F
import whisper
import opencc
from hyperpyyaml import load_hyperpyyaml
from huggingface_hub import snapshot_download
from g2pw import G2PWConverter

# 保持與原始程式一致，用於匯入自定義的 CosyVoice 模組與工具
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    replace_blank,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    split_paragraph,
)
from utils.word_utils import word_to_dataset_frequency, char2phn, always_augment_chars

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}/third_party/Matcha-TTS")

# 設定 PyTorch 只使用單執行緒，避免佔用過多資源
torch.set_num_threads(1)


class CustomCosyVoiceFrontEnd(CosyVoiceFrontEnd):
    """
    自訂版的 CosyVoiceFrontEnd，擴充並覆寫 text_normalize_new、frontend_zero_shot 及相關函式，
    以因應專案需求的前處理邏輯，包括：
    1. 中文與英文的正規化與標點符號處理
    2. 支援括號內外的特殊文本處理
    3. 分詞、音訊特徵提取等功能
    """

    def text_normalize_new(self, text: str, split: bool = False) -> str:
        """
        進階文字正規化：
        1. 根據中英文分別進行正規化
        2. 替換各種標點、去除括號
        3. 支援可選的 split=True/False，split=False 時直接輸出完成後的單一字串

        參數:
            text (str): 輸入要被正規化的文字
            split (bool): 是否要進行斷句切分（預設 False）
        回傳:
            (str): 完成正規化的字串
        """

        text = text.strip()

        def split_by_brackets(input_string: str):
            """
            使用正規表達式將輸入文字依照中括號切分成中括號內外兩部分。
            回傳 inside_brackets 以及 outside_brackets 兩個清單。
            """
            inside_brackets_list = re.findall(r"\[(.*?)\]", input_string)
            outside_brackets_list = re.split(r"\[.*?\]", input_string)
            # 移除因連續 bracket 產生的空字串
            outside_brackets_list = [part for part in outside_brackets_list if part]
            return inside_brackets_list, outside_brackets_list

        def text_normalize_no_split(text_segment: str, is_last: bool = False) -> str:
            """
            針對單一句段 (不包含括號) 進行正規化處理。包含：
            1. 中英文分流處理
            2. 去除多餘空白、替換標點、補足句尾
            3. 特殊數字拼寫

            參數:
                text_segment (str): 單一句段文字
                is_last (bool): 是否為最後一句，會影響是否補句尾標點
            回傳:
                (str): 正規化後的句段
            """
            text_segment = text_segment.strip()
            text_terminated = text_segment.endswith("。")

            if contains_chinese(text_segment):
                # 中文正規化
                if self.use_ttsfrd:
                    text_segment = self.frd.get_frd_extra_info(text_segment, "input")
                else:
                    text_segment = self.zh_tn_model.normalize(text_segment)

                # 如果本句未以句號結尾且也不是最後一句，就去掉最後一個句號
                if not text_terminated and not is_last:
                    text_segment = text_segment[:-1]

                text_segment = text_segment.replace("\n", "")
                text_segment = replace_blank(text_segment)
                text_segment = replace_corner_mark(text_segment)
                text_segment = text_segment.replace(".", "、")
                text_segment = text_segment.replace(" - ", "，")
                text_segment = remove_bracket(text_segment)
                text_segment = re.sub(r"[，,]+$", "。", text_segment)
            else:
                # 英文正規化與數字拼寫
                if self.use_ttsfrd:
                    text_segment = self.frd.get_frd_extra_info(text_segment, "input")
                else:
                    text_segment = self.en_tn_model.normalize(text_segment)

                text_segment = spell_out_number(text_segment, self.inflect_parser)

            return text_segment

        def join_interleaved(outside_list, inside_list):
            """
            將前面 split_by_brackets 得到的 outside 與 inside 交錯合併回來。
            """
            result = []
            for o, i in zip(outside_list, inside_list):
                result.append(o + f"[{i}]")
            if len(outside_list) > len(inside_list):
                result.append(outside_list[-1])
            return "".join(result)

        # 1. 先切分中括號內外
        inside_brackets, outside_brackets = split_by_brackets(text)

        # 2. 逐一正規化中括號外的文字
        for idx in range(len(outside_brackets)):
            segment_normalized = text_normalize_no_split(
                outside_brackets[idx],
                is_last=(idx == len(outside_brackets) - 1)
            )
            outside_brackets[idx] = segment_normalized

        # 3. 再將中括號的部分與正規化後的外部文字交錯合併
        text = join_interleaved(outside_brackets, inside_brackets)

        if not split:
            return text
        return text  # 如果未來要做進一步切分可在此擴充

    def frontend_zero_shot(self, tts_text: str, prompt_text: str, prompt_speech_16k: torch.Tensor) -> dict:
        """
        針對 Zero-Shot 模式，根據一段參考音檔 (prompt_speech_16k) 與其文字內容 (prompt_text)，
        輸出用於模型推論的特徵化輸入。

        參數:
            tts_text (str): 要生成語音的文字
            prompt_text (str): 參考音檔所對應的文字（同一位說話者）
            prompt_speech_16k (torch.Tensor): 16kHz 取樣率的參考音訊

        回傳:
            dict: 包含文本與音訊特徵，後續可直接丟給模型做推論使用
        """
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)

        # 音訊取樣率轉換 16k -> 22.05k
        prompt_speech_22050 = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=22050
        )(prompt_speech_16k)

        # 取出各種特徵: 例如音頻梅爾頻譜
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        embedding = self._extract_spk_embedding(prompt_speech_16k)

        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "prompt_text": prompt_text_token,
            "prompt_text_len": prompt_text_len,
            "llm_prompt_speech_token": speech_token,
            "llm_prompt_speech_token_len": speech_token_len,
            "flow_prompt_speech_token": speech_token,
            "flow_prompt_speech_token_len": speech_token_len,
            "prompt_speech_feat": speech_feat,
            "prompt_speech_feat_len": speech_feat_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_zero_shot_dual(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_speech_16k: torch.Tensor,
        flow_prompt_text: str,
        flow_prompt_speech_16k: torch.Tensor
    ) -> dict:
        """
        與 frontend_zero_shot 類似，但增加額外提示音檔 (flow_prompt_speech_16k) 與文本 (flow_prompt_text)，
        可用於多條件或更複雜的合成需求。

        參數:
            tts_text (str): 要合成的文字
            prompt_text (str): 第一路提示音訊對應的文字
            prompt_speech_16k (torch.Tensor): 第一路 16kHz 提示音訊
            flow_prompt_text (str): 第二路提示音訊對應的文字
            flow_prompt_speech_16k (torch.Tensor): 第二路 16kHz 提示音訊

        回傳:
            dict: 包含文本與音訊特徵的字典，用於模型推論
        """
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        flow_prompt_text_token, flow_prompt_text_token_len = self._extract_text_token(flow_prompt_text)

        flow_prompt_speech_22050 = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=22050
        )(flow_prompt_speech_16k)

        speech_feat, speech_feat_len = self._extract_speech_feat(flow_prompt_speech_22050)
        flow_speech_token, flow_speech_token_len = self._extract_speech_token(flow_prompt_speech_16k)

        # 這裡的設計看來是重用 flow_speech_token 來作為 llm_prompt_speech_token
        speech_token = flow_speech_token.clone()
        speech_token_len = flow_speech_token_len.clone()

        embedding = self._extract_spk_embedding(prompt_speech_16k)
        flow_embedding = embedding.clone()

        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "prompt_text": prompt_text_token,
            "prompt_text_len": prompt_text_len,
            "llm_prompt_speech_token": speech_token,
            "llm_prompt_speech_token_len": speech_token_len,
            "flow_prompt_speech_token": flow_speech_token,
            "flow_prompt_speech_token_len": flow_speech_token_len,
            "prompt_speech_feat": speech_feat,
            "prompt_speech_feat_len": speech_feat_len,
            "llm_embedding": embedding,
            "flow_embedding": flow_embedding,
        }
        return model_input


class CustomCosyVoiceModel(CosyVoiceModel):
    """
    自訂的 CosyVoiceModel 用於載入並管理三個模組：LLM、Flow 以及 HiFT，
    包含模型的 load 與 inference 流程，整合整個文字到語音的處理步驟。
    """

    def __init__(
        self,
        llm: torch.nn.Module,
        flow: torch.nn.Module,
        hift: torch.nn.Module
    ):
        """
        建構子，接受三個子模型的類別實例，但還未載入權重。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = llm
        self.flow = flow
        self.hift = hift

    def load(self, llm_model: str, flow_model: str, hift_model: str) -> None:
        """
        載入預訓練好的 LLM、Flow 與 HiFT 模型檔，並移到指定的裝置 (CPU 或 GPU)。
        """
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
        self.llm.to(self.device).eval()

        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()

        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        flow_embedding: torch.Tensor,
        llm_embedding: torch.Tensor = torch.zeros(0, 192),
        prompt_text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        prompt_text_len: torch.Tensor = torch.zeros(1, dtype=torch.int32),
        llm_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token_len: torch.Tensor = torch.zeros(1, dtype=torch.int32),
        flow_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token_len: torch.Tensor = torch.zeros(1, dtype=torch.int32),
        prompt_speech_feat: torch.Tensor = torch.zeros(1, 0, 80),
        prompt_speech_feat_len: torch.Tensor = torch.zeros(1, dtype=torch.int32),
    ) -> dict:
        """
        呼叫 LLM 產生音素序列，再使用 Flow 模組將音素序列轉成 Mel 頻譜，
        最後透過 HiFT 將 Mel 頻譜轉換為最終的時間域波形。

        回傳:
            dict: 包含 tts_speech (音檔) 的字典
        """
        # LLM 推論：將文字 -> 音素序列
        tts_speech_token = self.llm.inference(
            text=text.to(self.device),
            text_len=text_len.to(self.device),
            prompt_text=prompt_text.to(self.device),
            prompt_text_len=prompt_text_len.to(self.device),
            prompt_speech_token=llm_prompt_speech_token.to(self.device),
            prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
            embedding=llm_embedding.to(self.device),
            beam_size=1,
            sampling=25,
            max_token_text_ratio=30,
            min_token_text_ratio=3,
        )

        # Flow 推論：將 LLM 產生的音素序列 -> Mel 頻譜
        tts_mel = self.flow.inference(
            token=tts_speech_token,
            token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(self.device),
            prompt_token=flow_prompt_speech_token.to(self.device),
            prompt_token_len=flow_prompt_speech_token_len.to(self.device),
            prompt_feat=prompt_speech_feat.to(self.device),
            prompt_feat_len=prompt_speech_feat_len.to(self.device),
            embedding=flow_embedding.to(self.device),
        )

        # HiFT 推論：將 Mel 頻譜 -> 時域波形
        tts_speech = self.hift.inference(mel=tts_mel).cpu()

        # 清空 GPU 快取以節省顯存
        torch.cuda.empty_cache()

        return {"tts_speech": tts_speech}


class CustomCosyVoice:
    """
    封裝整個 Zero-Shot TTS 流程的主類別，包括：
    1. 載入 CosyVoice 前處理與模型
    2. 根據不同模式 (SFT / Zero-Shot / Dual 等) 執行語音合成
    """

    def __init__(self, model_dir: str):
        """
        初始化:
        1. 檢查 model_dir 是否存在或需要從 HuggingFace Hub 下載
        2. 讀取 cosyvoice.yaml
        3. 建立 CustomCosyVoiceFrontEnd 與 CustomCosyVoiceModel 並載入
        """
        instruct = False
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        print("Model path:", model_dir)

        self.model_dir = model_dir

        # 讀取超參數
        with open(f"{model_dir}/cosyvoice.yaml", "r") as f:
            configs = load_hyperpyyaml(f)

        # 自訂 FrontEnd
        self.frontend = CustomCosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            model_dir,
            f"{model_dir}/campplus.onnx",
            f"{model_dir}/speech_tokenizer_v1.onnx",
            f"{model_dir}/spk2info.pt",
            instruct,
            configs["allowed_special"],
        )

        # 建立模型並載入參數
        self.model = CosyVoiceModel(
            configs["llm"], configs["flow"], configs["hift"]
        )
        self.model.load(
            f"{model_dir}/llm.pt",
            f"{model_dir}/flow.pt",
            f"{model_dir}/hift.pt",
        )
        del configs  # 釋放暫存記憶體

    def list_avaliable_spks(self):
        """
        列出預訓練模型中可用的 speaker IDs (若有提供)。
        """
        return list(self.frontend.spk2info.keys())

    def inference_sft(self, tts_text: str, spk_id: str) -> dict:
        """
        已微調 (SFT) 模式，適用於模型本身訓練時就已經存在多種預設說話人 (spk_id)。
        """
        tts_speeches = []
        # split=True 代表會自動切分長句子
        for seg_text in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_sft(seg_text, spk_id)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output["tts_speech"])
        return {"tts_speech": torch.concat(tts_speeches, dim=1)}

    def inference_zero_shot(self, tts_text: str, prompt_text: str, prompt_speech_16k: torch.Tensor) -> dict:
        """
        Zero-Shot 模式，根據一小段參考音檔 (prompt_speech_16k) 與文字 (prompt_text)，
        讓模型模仿該說話者進行語音合成。
        """
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        tts_speeches = []
        for seg_text in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_zero_shot(
                seg_text, prompt_text, prompt_speech_16k
            )
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output["tts_speech"])
        return {"tts_speech": torch.concat(tts_speeches, dim=1)}

    def inference_zero_shot_no_unit_condition_no_normalize(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_speech_16k: torch.Tensor,
        flow_prompt_text: str = None,
        flow_prompt_speech_16k: torch.Tensor = None
    ) -> dict:
        """
        與 inference_zero_shot 類似，但不進行額外的音素條件 (unit condition) 處理，也跳過常規化流程。
        適合要自行先行處理語句或測試多路提示音檔的情境。
        """
        if flow_prompt_text is None:
            flow_prompt_text = prompt_text
        if flow_prompt_speech_16k is None:
            flow_prompt_speech_16k = prompt_speech_16k

        tts_speeches = []
        # 依據中文常見標點符號進行簡單斷句
        for seg_text in re.split(r"(?<=[？！。.?!])\s*", tts_text):
            if not len(seg_text.strip()):
                continue
            model_input = self.frontend.frontend_zero_shot_dual(
                seg_text,
                prompt_text,
                prompt_speech_16k,
                flow_prompt_text,
                flow_prompt_speech_16k,
            )
            # 移除 llm_prompt_speech_token 的內容
            model_input["llm_prompt_speech_token"] = model_input["llm_prompt_speech_token"][:, :0]
            model_input["llm_prompt_speech_token_len"][0] = 0

            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output["tts_speech"])

        return {"tts_speech": torch.concat(tts_speeches, dim=1)}

    def inference_zero_shot_no_normalize(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_speech_16k: torch.Tensor
    ) -> dict:
        """
        與 inference_zero_shot 類似，但不另外對 tts_text 與 prompt_text 做正規化；
        若使用者外部已經做好清洗或正規化，可直接輸入。
        """
        tts_speeches = []
        for seg_text in re.split(r"(?<=[？！。.?!])\s*", tts_text):
            if not len(seg_text.strip()):
                continue
            print("Synthesizing:", seg_text)
            model_input = self.frontend.frontend_zero_shot(
                seg_text,
                prompt_text,
                prompt_speech_16k
            )
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output["tts_speech"])
        return {"tts_speech": torch.concat(tts_speeches, dim=1)}


def transcribe_audio(audio_file: str) -> str:
    """
    使用 (transformers) Whisper-base 做語音識別，並將簡體中文結果轉為繁體中文。
    """
    from transformers import pipeline
    whisper_asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    result = whisper_asr(audio_file)

    converter = opencc.OpenCC("s2t")
    traditional_text = converter.convert(result["text"])
    return traditional_text


def get_bopomofo_rare(text: str, converter: G2PWConverter) -> str:
    """
    為較罕見的中文字元加上注音標記 (Bopomofo)。若該字在資料集中頻次較低，就會在後方加上 [:注音]。
    另外，若同一字有多個可能注音，依照特定條件 (常用 / 非常用) 來判定是否需要補注音。
    """
    # converter(...) 回傳一個 (text, phoneme) 配對列表
    phoneme_list = converter(text)
    text_with_bopomofo = [x for x in zip(list(text), phoneme_list[0])]
    reconstructed_text = ""

    for i, (char, bopomo) in enumerate(text_with_bopomofo):
        next_char = text_with_bopomofo[i+1][0] if i < len(text_with_bopomofo) - 1 else None

        # 若此字在資料集出現頻次很低且後面一個字不是 "[" 才加注音
        if word_to_dataset_frequency[char] < 500 and bopomo is not None and next_char != "[":
            reconstructed_text += f"{char}[:{bopomo}]"

        elif len(char2phn[char]) >= 2:
            # 有多音，並且不是最常見的音或該字屬於 always_augment_chars
            if (bopomo != char2phn[char][0] and
                (word_to_dataset_frequency[char] < 10000 or char in always_augment_chars) and
                next_char != "["):
                reconstructed_text += f"{char}[:{bopomo}]"
            else:
                reconstructed_text += char
        else:
            reconstructed_text += char

    return reconstructed_text


def parse_transcript(text: str, end_time: float) -> tuple:
    """
    解析帶有時間標記的文字內容。
    <|start_time|>文字<|end_time|> 的結構。
    用於根據時間範圍篩選出對應文字，最後回傳一段合併後的文字內容與實際起始時間。
    """
    pattern = r"<\|(\d+\.\d+)\|>([^<]+)<\|(\d+\.\d+)\|>"
    matches = re.findall(pattern, text)
    parsed_output = [(float(s), float(e), content.strip()) for s, content, e in matches]

    # 根據條件過濾不需要的片段
    count_zero = 0
    for i in range(len(parsed_output)):
        if parsed_output[i][0] == 0:
            count_zero += 1
        if count_zero >= 2:
            parsed_output = parsed_output[:i]
            break

    for i in range(len(parsed_output)):
        if parsed_output[i][0] >= end_time:
            parsed_output = parsed_output[:i]
            break

    for i in range(len(parsed_output)):
        if parsed_output[i][0] < end_time - 15:
            continue
        else:
            parsed_output = parsed_output[i:]
            break

    start = parsed_output[0][0]
    merged_text = "".join([p[2] for p in parsed_output])
    return merged_text, start


def single_inference(
    speaker_prompt_audio_path: str,
    content_to_synthesize: str,
    output_path: str,
    cosyvoice: CustomCosyVoice,
    bopomofo_converter: G2PWConverter,
    speaker_prompt_text_transcription: str = None
) -> None:
    """
    單次推論的主要函式：
    1. 載入提示音訊 (prompt_speech_16k)
    2. 若無提供提示文字，則使用 Whisper-base 做語音辨識
    3. 進行文字的正規化與注音化
    4. 使用 CustomCosyVoice 模型執行 zero-shot 語音合成
    5. 寫出合成後的音訊檔案到指定路徑
    """

    # 1. 載入參考音檔 (16kHz)
    prompt_speech_16k = load_wav(speaker_prompt_audio_path, 16000)

    # 2. 若無提供轉錄文本，透過 Whisper-base 自動產生
    if speaker_prompt_text_transcription:
        speaker_prompt_text = speaker_prompt_text_transcription
    else:
        speaker_prompt_text = transcribe_audio(speaker_prompt_audio_path)

    # 3. 對提示音檔文字與合成內容做正規化 + 注音標記
    speaker_prompt_text = cosyvoice.frontend.text_normalize_new(speaker_prompt_text, split=False)
    content_to_synthesize = cosyvoice.frontend.text_normalize_new(content_to_synthesize, split=False)

    speaker_prompt_text_bopomo = get_bopomofo_rare(speaker_prompt_text, bopomofo_converter)
    content_to_synthesize_bopomo = get_bopomofo_rare(content_to_synthesize, bopomofo_converter)

    print("Speaker prompt transcription (bopomofo):", speaker_prompt_text_bopomo)
    print("Content to be synthesized (bopomofo):", content_to_synthesize_bopomo)

    # 4. 推論
    start_time = time.time()
    output_dict = cosyvoice.inference_zero_shot_no_normalize(
        content_to_synthesize_bopomo,
        speaker_prompt_text_bopomo,
        prompt_speech_16k
    )
    end_time = time.time()

    # 5. 結果輸出
    print("Elapsed time:", end_time - start_time)
    print("Generated audio length:", output_dict["tts_speech"].shape[1] / 22050, "seconds")

    torchaudio.save(output_path, output_dict["tts_speech"], 22050)
    print(f"Generated voice saved to {output_path}")


def main():
    """
    主程式入口：
    1. 處理命令列參數
    2. 建立自訂的 CosyVoice 實例
    3. 依照參數執行合成並輸出結果
    """
    parser = argparse.ArgumentParser(description="Run BreezyVoice text-to-speech with custom inputs")

    parser.add_argument(
        "--content_to_synthesize",
        type=str,
        required=True,
        help="指定要合成的語音文字"
    )
    parser.add_argument(
        "--speaker_prompt_audio_path",
        type=str,
        required=True,
        help="指定參考說話者音檔路徑 (16kHz)"
    )
    parser.add_argument(
        "--speaker_prompt_text_transcription",
        type=str,
        required=False,
        help="選擇性，提供提示音檔的文字轉錄，若不提供將使用 Whisper 自動辨識"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="results/output.wav",
        help="輸出合成音檔的路徑與檔名"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="MediaTek-Research/BreezyVoice-300M",
        help="指定預訓練模型所在的資料夾或 HuggingFace Hub 路徑"
    )

    args = parser.parse_args()

    # 建立自訂 CosyVoice 與注音轉換器
    cosyvoice = CustomCosyVoice(args.model_path)
    bopomofo_converter = G2PWConverter()

    # 進行單次語音合成推論
    single_inference(
        speaker_prompt_audio_path=args.speaker_prompt_audio_path,
        content_to_synthesize=args.content_to_synthesize,
        output_path=args.output_path,
        cosyvoice=cosyvoice,
        bopomofo_converter=bopomofo_converter,
        speaker_prompt_text_transcription=args.speaker_prompt_text_transcription
    )


if __name__ == "__main__":
    main()
