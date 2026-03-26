---
name: silicon-detect
description: Detect language of text or audio using Apple's on-device models. Useful for multilingual content processing.
argument-hint: "<text>"
---

# /silicon-detect

Detect the language of text using Apple's NaturalLanguage framework (on-device, no API key needed).

```bash
python3 -c "
import NaturalLanguage
recognizer = NaturalLanguage.NLLanguageRecognizer.alloc().init()
recognizer.processString_('${ARGS}')
lang = recognizer.dominantLanguage()
hypotheses = recognizer.languageHypothesesWithMaximum_(5)
print(f'Dominant: {lang}')
print('Top hypotheses:')
for lang, conf in sorted(hypotheses.items(), key=lambda x: -x[1]):
    print(f'  {lang}: {conf:.3f}')
" 2>&1
```

If Python NaturalLanguage bindings aren't available:
```bash
# Use Swift via command line
swift -e "
import NaturalLanguage
let r = NLLanguageRecognizer()
r.processString(\"${ARGS}\")
print(r.dominantLanguage?.rawValue ?? \"unknown\")
" 2>&1
```

Report the detected language with confidence scores.
