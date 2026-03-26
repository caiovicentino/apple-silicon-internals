---
name: silicon-ocr
description: Extract text from images using Apple's on-device OCR engine (TextRecognition/Vision). Supports 20+ languages including Chinese, Japanese, Korean, Arabic.
argument-hint: "<image_path>"
---

# /silicon-ocr

Extract text from an image using Apple's on-device OCR via the Vision framework.

```bash
# Use the public Vision API via Python (faster to invoke than building ObjC)
python3 -c "
import Vision, Quartz, sys
path = '${ARGS}'
image = Quartz.CIImage.imageWithContentsOfURL_(Quartz.NSURL.fileURLWithPath_(path))
handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(image, None)
request = Vision.VNRecognizeTextRequest.alloc().init()
request.setRecognitionLevel_(1)  # 1 = accurate
handler.performRequests_error_([request], None)
for obs in request.results():
    print(obs.topCandidates_(1)[0].string())
" 2>&1
```

If Python Vision bindings aren't available, fall back to the `screencapture` + `shortcuts` approach:
```bash
shortcuts run "Extract Text from Image" -i "${ARGS}" 2>/dev/null || echo "Use /silicon-scan TextRecognition to explore the OCR API surface"
```

Report the extracted text. Note any confidence issues or partial recognition.
