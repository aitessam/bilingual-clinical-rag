# Data

## Important: synthetic data only

Everything under `raw/` in this folder is **entirely synthetic**, authored specifically for this project's bilingual RAG demo. It does **not** contain any real patient data, and no patient names are used anywhere (notes refer only to "the patient" or "a XX-year-old male/female patient").

The notes are styled after the format of the public MTSamples medical transcription reports (chief complaint, history of present illness, past medical history, medications, allergies, physical examination, assessment, plan), but the content itself is fully fabricated for this project. Nothing here was scraped or copied from MTSamples or any other real source.

## Layout

```
raw/
  en/   English notes, note_001.txt to note_032.txt
  ar/   Arabic notes, note_001.txt to note_032.txt
processed/  Reserved for chunked/indexed output from the ingestion pipeline (currently empty)
```

Files are paired by name: `en/note_NNN.txt` and `ar/note_NNN.txt` describe the same synthetic case in each language, so they can be used to test cross-lingual retrieval.

The Arabic versions are professional-register adaptations, not literal word-for-word translations, but they preserve the same clinical facts (age, sex, vital signs, medications, diagnosis, plan) as their English counterpart.

## Specialty breakdown (32 notes total, 4 per specialty)

| Notes | Specialty |
|---|---|
| note_001 to note_004 | Cardiology |
| note_005 to note_008 | Orthopedics |
| note_009 to note_012 | Gastroenterology |
| note_013 to note_016 | Neurology |
| note_017 to note_020 | Endocrinology |
| note_021 to note_024 | Pulmonology |
| note_025 to note_028 | Nephrology |
| note_029 to note_032 | Dermatology |
