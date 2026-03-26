---
name: silicon-ner
description: Extract named entities (people, places, organizations) and detect data types (dates, addresses, phone numbers, emails, URLs) from text using Apple's on-device NLP.
argument-hint: "<text>"
---

# /silicon-ner

Run Named Entity Recognition and Data Detection on text using Apple's on-device NaturalLanguage and NSDataDetector APIs.

## NER (people, places):
```bash
swift -e '
import NaturalLanguage
let text = "${ARGS}"
let tagger = NLTagger(tagSchemes: [.nameType])
tagger.string = text
print("=== Named Entities ===")
tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameType) { tag, range in
    if let tag = tag, tag != .otherWord, tag != .whitespace, tag != .punctuation, tag != .sentenceTerminator {
        print("  [\(tag.rawValue)] \"\(text[range])\"")
    }
    return true
}
'
```

## Data Detection (dates, addresses, phones, emails, URLs):
```bash
swift -e '
import Foundation
let text = "${ARGS}"
let detector = try! NSDataDetector(types: NSTextCheckingAllTypes)
let matches = detector.matches(in: text, range: NSRange(location: 0, length: text.utf16.count))
print("=== Data Detected ===")
for m in matches {
    let range = Range(m.range, in: text)!
    let typeNames: [UInt64: String] = [1:"Date", 16:"Address", 2048:"Phone", 32:"Link/Email", 4096:"Transit"]
    let typeName = typeNames[m.resultType.rawValue] ?? "Type(\(m.resultType.rawValue))"
    print("  [\(typeName)] \"\(text[range])\"")
}
'
```

Present both results together: entities found and structured data detected.
