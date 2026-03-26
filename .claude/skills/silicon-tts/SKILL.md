---
name: silicon-tts
description: Generate speech audio from text using Apple's on-device TTS. Supports 184 voices across dozens of languages including pt-BR.
argument-hint: "<text> [language] [output_path]"
---

# /silicon-tts

Generate an audio file from text using Apple's AVSpeechSynthesizer (on-device, no API key).

Write and run this Swift script:

```bash
ARGS_TEXT="${ARGS}"
# Default language pt-BR, default output /tmp/tts_output.wav
cat << 'SWIFT' > /tmp/tts_script.swift
import AVFoundation
import Foundation

let args = CommandLine.arguments
let text = args.count > 1 ? args[1] : "Texto de exemplo"
let lang = args.count > 2 ? args[2] : "pt-BR"
let output = args.count > 3 ? args[3] : "/tmp/tts_output.wav"

let synth = AVSpeechSynthesizer()
let utterance = AVSpeechUtterance(string: text)
utterance.voice = AVSpeechSynthesisVoice(language: lang)
utterance.rate = 0.48

let outputURL = URL(fileURLWithPath: output)
var audioFile: AVAudioFile?

synth.write(utterance) { buffer in
    guard let pcmBuffer = buffer as? AVAudioPCMBuffer, pcmBuffer.frameLength > 0 else { return }
    if audioFile == nil {
        audioFile = try? AVAudioFile(forWriting: outputURL, settings: pcmBuffer.format.settings)
    }
    try? audioFile?.write(from: pcmBuffer)
}

RunLoop.current.run(until: Date(timeIntervalSinceNow: 60))

if FileManager.default.fileExists(atPath: output) {
    let size = try! FileManager.default.attributesOfItem(atPath: output)[.size] as! Int
    print("Generated: \(output) (\(size / 1024) KB)")
} else {
    print("Failed")
}
SWIFT
swift /tmp/tts_script.swift "$ARGS_TEXT"
```

To also play the audio after generating:
```bash
afplay /tmp/tts_output.wav &
```

To list available voices for a language:
```bash
swift -e 'import AVFoundation; AVSpeechSynthesisVoice.speechVoices().filter{$0.language.starts(with:"pt")}.forEach{print("\($0.name): \($0.language)")}'
```
