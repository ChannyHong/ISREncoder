# ISREncoder

This is the project page for the Tensorflow implementation of the paper, "Unsupervised Interlingual Semantic Representations from Sentence Embeddings for Zero-Shot Cross-Lingual Transfer", to be presented at the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20).

Copyright 2020 Superb AI, Inc.\
The code, cache files, and the models are all released under the Apache 2.0 license.\
Authors: Channy Hong, Jaeyeon Lee, Jung Kwon Lee.

Paper: (AAAI-20 & arXiv links coming soon!)\
Overview blog post: [Medium](https://medium.com/superb-ai/training-non-english-nlp-models-with-english-training-data-664bbd260681)

---

## ISR Encoder Training

Script for training the ISR Encoder. Requires monolingual corpora cache files for training.

**Prerequisites**:

The following cache files saved in the 'data_dir' directory:
- Monolingual corpora sentences cache files, as mc_##.npy (e.g. mc_en.npy) where ## corresponds to [ISO 639-1 Code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) of each 'train_languages'; refer to Parsing and Caching Scripts section below.
- (If do_mid_train_eval) XNLI dev examples cache file, as [DEV.npy](https://drive.google.com/uc?export=download&id=1VOZqXGrLRjVbSmf-wB9ETrBtGfQPNZ7L); refer to Parsing and Caching Scripts section below.

(If do_mid_train_eval,) The following model files in the 'mid_train_eval_nli_model_path' directory (the trailing 'nli_solver' is the model name and not part of the directory):
-  English NLI solver model files, as [nli_solver.meta](https://drive.google.com/uc?export=download&id=1RroiNFZVxap9FPCwSkS78CiXfoxHkgzw), [nli_solver.index](https://drive.google.com/uc?export=download&id=1sptgnDG8lhj415OVnjHt25peORMy8a8E), and [nli_solver.data-00000-of-00001](https://drive.google.com/uc?export=download&id=1nDtQFFOM7EnA8sX5viyXcrwkWoGsw1JE) (note that 'mid_train_eval_nli_target_language' should be fixed as English when using this NLI solver).
```
python train_isr.py \
  --data_dir=data \
  --output_dir=outputs/isr_training_model \
  --train_languages=English,Spanish,German,Chinese,Arabic \
  --embedding_size=768 \
  --train_batch_size=32 \
  --Dis_Gen_train_ratio=10 \
  --Dis_learning_rate=0.00001 \
  --Gen_learning_rate=0.00001 \
  --lambda_Dis_cls=0.0 \
  --lambda_Dis_gp=1.0 \
  --lambda_Gen_cls=0.0 \
  --lambda_Gen_rec=1.0 \
  --lambda_Gen_isr=1.0 \
  --beta1=0.5 \
  --beta2=0.999 \
  --num_train_epochs=100 \
  --save_checkpoints_steps=5000 \
  --log_losses=True
  --do_mid_train_eval=True \
  --run_mid_train_eval_steps=5000 \
  --mid_train_eval_nli_target_language=English \
  --mid_train_eval_nli_model_path=nli_solver_path/nli_solver
```

---

## Classifier Training

Code for training a classifier on top of fixed ISR Encoder. Requires NLI training examples (mostly available in high-resource language, i.e. English) for training.

**Prerequisites**:

The following cache files saved in the 'data_dir' directory:
- NLI training examples cache file(s), as bse_##.npy (e.g. bse_en.npy) where ## corresponds to [ISO 639-1 Code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) of each 'xnli_train_languages'; refer to Parsing and Caching Scripts section below. Theoretically, NLI training examples from multiple languages can be used jointly from training the classifier on top of ISR (while the underlying assumption is that only English training examples are widely available currently).
- (if do_mid_train_eval) XNLI dev examples cache file, as [DEV.npy](https://drive.google.com/uc?export=download&id=1VOZqXGrLRjVbSmf-wB9ETrBtGfQPNZ7L).

The following files in the 'isr_encoder_dir' directory:
- The ISR Encoder model files outputted from the ISR Encoder Training section above, as isr_encoder.meta, isr_encoder.index, isr_encoder.data-00000-of-00001. Alternatively, the ISR Encoder trained during our study can downloaded here: [isr_encoder.meta](https://drive.google.com/uc?export=download&id=1LJ9l-r2OoBPAt-7kNTWr34W7Et-gzhFN), [isr_encoder.index](https://drive.google.com/uc?export=download&id=1PIOseaAo37SeKe7_lbSGqQdFI-4h0ywH), and [isr_encoder.data-00000-of-00001](https://drive.google.com/uc?export=download&id=1Y0IyQOKZsknMEhQTzdGFyj9zFDtRi2PW).
- The language reference file, as language_reference.json. The language reference file corresponding to our study's ISR Encoder can be downloaded here: [language_refernce.json](https://drive.google.com/uc?export=download&id=1Owm6Hv6KKE1NLGhTtgAGINfZc94LHYA_)

```
python train_classifier.py \
  --data_dir=data \
  --isr_encoder_dir=isr_encoder_dir \
  --isr_encoder_name=isr_encoder \
  --output_dir=outputs/custom_output_model_name \
  --xnli_train_languages=English \
  --embedding_size=768 \
  --train_batch_size=32 \
  --dropout_rate=0.5 \
  --learning_rate=0.00005 \
  --beta1=0.9 \
  --beta2=0.999 \
  --num_train_epochs=100 \
  --save_checkpoints_steps=5000 \
  --log_losses=True \
  --do_mid_train_eval=True \
  --mid_train_xnli_eval_languages=English,Spanish,German,Chinese,Arabic \
  --run_mid_train_eval_steps=5000 \
  --mid_train_eval_batch_size=32
```

---

## Parsing and Caching Scripts

### Producing a monolingual corpora cache file from Wikipedia dump

**1. Download the [Wikipedia dump](https://dumps.wikimedia.org/) of the language of interest (.XML file).**

**2. Use [WikiExtractor](https://github.com/attardi/wikiextractor) to extract and clean text from the XML file, outputting a file (e.g. wiki_00) in the "AA" folder within the 'output' directory. The "100G" 'bytes' parameter in our sample usage is to ensure that only 1 file is outputted (rather than broken up into multiple)**:

**Prerequisites**:
- The downloaded dump file (e.g. en_dump.xml) in the current directory.
```
python WikiExtractor.py \
 --output=en_extracted \
 --bytes=100G \
en_dump.xml
```

**3. Run mc_custom_extraction.py on once-extracted file to perform custom extraction and cleanup to output a .txt file.**

**Prerequisites**:
- The once-extracted dump file (e.g. wiki_00) in the 'source_file_path' directory (the trailing source file name is not part of the directory and must match the dump file name).

```
python mc_custom_extraction.py \
  --source_file_path=once_extracted/wiki_00 \
  --output_dir=custom_extracted \
  --language=en \
  --char_count_lower_bound=4 \
  --char_count_upper_bound=385 \
  --output_num_examples=392702
```

The monolingual corpora .txt files used in our study can be downloaded here:\
[mc_en.txt](https://drive.google.com/uc?export=download&id=1SkZKzfMY2X5_1XNOvfIE5RSjiA34ec6z)\
[mc_es.txt](https://drive.google.com/uc?export=download&id=1LsoXQgGGp5n_Ks1sFO4AHTe9UH6QUYA7)\
[mc_de.txt](https://drive.google.com/uc?export=download&id=1Mz-wxBcgkMKruep59LB3RgWXnYs15h3_)\
[mc_zh.txt](https://drive.google.com/uc?export=download&id=1jXYRLgox3R_K46uDhOlhzUEY0gYax1pR)\
[mc_ar.txt](https://drive.google.com/uc?export=download&id=1NsRpJvjcjhx4lYc6Of4JfsXVPZi92LW2)

**4. Run bse_cache.py to produce cache files.**

**Prerequisites**:
- [bert-as-service](https://github.com/hanxiao/bert-as-service) installed.
- [BERT-Base, Multilingual Cased model](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) (refer to [BERT Multilingual GitHub page](https://github.com/google-research/bert/blob/master/multilingual.md) for more details) saved in the 'bert_dir' directory.
- The custom extracted .txt file in the 'data_dir' directory, as mc_##.txt where ## corresponds to [ISO 639-1 Code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) of the text.

```
python bse_cache.py \
  --data_dir=custom_extracted \
  --language=English \
  --data_type=mc \
  --output_dir=bse_cache_files \
  --bert_dir=../pretrained_models/multi_cased_L-12_H-768_A-12
```

The monolingual corpora cache files used in our study can be downloaded here:\
[mc_en.npy](https://drive.google.com/uc?export=download&id=1LArWH8bU2sL0o-Ih2y2re2osxRGnv8bJ)\
[mc_es.npy](https://drive.google.com/uc?export=download&id=1_PEgGCv7e4YJKWhDBhKseHc9GyYpF2Cj)\
[mc_de.npy](https://drive.google.com/uc?export=download&id=1lcUBEKOry8JscOsM4P3NKeWI0ZZfj8i4)\
[mc_zh.npy](https://drive.google.com/uc?export=download&id=1JZSHRXR_JAelUspEgl2OAS4irIuhdGC2)\
[mc_ar.npy](https://drive.google.com/uc?export=download&id=1VfUl8B0c0o1KqtxHHLXNlGELw9djOuhX)

### Producing a NLI examples cache file from XNLI dataset

**1. Download the [XNLI dev and test examples](https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip) (xnli.dev.tsv and xnli.test.tsv) from the [XNLI project page](https://www.nyu.edu/projects/bowman/xnli/). Also download the [XNLI machine translated training examples](https://www.nyu.edu/projects/bowman/xnli/XNLI-MT-1.0.zip), which includes the original English MNLI training examples (as multinli.train.en.tsv).**

**2. Run bse_cache.py to produce cache files.**

#### English MNLI training examples

**Prerequisites**:
- The English MNLI training examples file in the 'data_dir' directory, as multinli.train.en.tsv.
```
python bse_cache.py \
  --data_dir=xnli_data \
  --language=English \
  --data_type=mnli \
  --output_dir=bse_cache_files \
  --bert_dir=../pretrained_models/multi_cased_L-12_H-768_A-12
```
The English MNLI training examples cache file used in our study can be downloaded here: [bse_en.npy](https://drive.google.com/uc?export=download&id=1dzOhSUraOtwhSjReoQhISsMeAnqpXhS5)

#### XNLI dev examples
**Prerequisites**:
- The XNLI dev examples file in the 'data_dir' directory, as xnli.dev.tsv.
```
python bse_cache.py \
  --data_dir=xnli_data \
  --data_type=dev \
  --output_dir=bse_cache_files \
  --bert_dir=../pretrained_models/multi_cased_L-12_H-768_A-12
```
The XNLI dev examples cache file used in our study can be downloaded here: [DEV.npy](https://drive.google.com/uc?export=download&id=1VOZqXGrLRjVbSmf-wB9ETrBtGfQPNZ7L)

#### XNLI test examples
**Prerequisites**:
- The XNLI test examples file in the 'data_dir' directory, as xnli.test.tsv.
```
python bse_cache.py \
  --data_dir=xnli_data \
  --data_type=test \
  --output_dir=bse_cache_files \
  --bert_dir=../pretrained_models/multi_cased_L-12_H-768_A-12
```
The XNLI test examples cache file used in our study can be downloaded here: [TEST.npy](https://drive.google.com/uc?export=download&id=12u5oTmpGZ0hpZTNyY_ABSQRTvU6U--rm)