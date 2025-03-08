python3 single_inference.py \
    --speaker_prompt_audio_path /app/data/example.wav \
    --speaker_prompt_text_transcription "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。" \
    --content_to_synthesize "Rust 被認定是一個有生產力的工具，讓能力不均的大型系統程式設計團隊能夠協同開發。底層程式碼容易產生難以察覺的錯誤，在多數其他語言中，只能靠大量的測試、以及經驗豐富的開發者小心翼翼地審核程式碼，才能找出它們。而在 Rust 中，編譯器扮演著守門員的角色阻擋這些難以捉摸的程式錯誤，包含並行（concurrency）的錯誤。透過與編譯器一同合作，開發團隊可以將他們的時間專注在程式邏輯，而不是成天追著錯誤跑。" \
    --output_path /app/results/test-$(date +'%Y%m%d').wav \
    --model_path /app/model


python3 single_inference.py \
    --speaker_prompt_audio_path /app/data/rust-test-2.wav \
    --speaker_prompt_text_transcription "$(cat /app/data/rust-test.txt)" \
    --content_to_synthesize "$(cat /app/data/rust-test-2.txt)" \
    --output_path "/app/results/rust-test-$(date +'%Y%m%d').wav" \
    --model_path /app/model

python3 single_inference.py \
    --speaker_prompt_audio_path /app/data/example.wav \
    --speaker_prompt_text_transcription "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。" \
    --content_to_synthesize "$(cat /app/data/text-1.txt)" \
    --output_path /app/results/text-2.wav \
    --model_path /app/model

python3 single_inference.py \
    --speaker_prompt_audio_path /app/sample/rust-test-2.wav \
    --speaker_prompt_text_transcription "$(cat /app/sample/rust-test.txt)" \
    --content_to_synthesize "$(cat /app/sample/clone.txt)" \
    --output_path "/app/sample/clone-$(date +'%Y%m%d').wav" \
    --model_path /app/model

python3 single_inference.py \
    --speaker_prompt_audio_path /app/sample/sample.wav \
    --speaker_prompt_text_transcription "$(cat /app/sample/sample.md)" \
    --content_to_synthesize "$(cat /app/sample/clone.md)" \
    --output_path /app/sample/clone-$(date +'%Y%m%d').wav \
    --model_path /app/model

python3 single_inference.py \
    --speaker_prompt_audio_path /app/sample/sample.wav \
    --speaker_prompt_text_transcription "$(cat /app/sample/sample.md)" \
    --content_to_synthesize "$(cat /app/sample/target-1.md)" \
    --output_path /app/sample/target-1-$(date +'%Y%m%d').wav \
    --model_path /app/model

python3 single_inference.py \
    --speaker_prompt_audio_path /app/data/example.wav \
    --speaker_prompt_text_transcription "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。" \
    --content_to_synthesize "$(cat /app/sample/clone.md)" \
    --output_path /app/sample/clone-$(date +'%Y%m%d').wav \
    --model_path /app/model