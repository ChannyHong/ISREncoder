

# ISREncoder

This is the project page for the Tensorflow implementation of the paper, "Unsupervised Interlingual Semantic Representations from Sentence Embeddings for Zero-Shot Cross-Lingual Transfer", to be presented at the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20).

Copyright 2020 Superb AI, Inc.\
The code, cache files, and the models are all released under the Apache 2.0 license.\
Authors: Channy Hong, Jaeyeon Lee, Jung Kwon Lee.

Paper: (AAAI-20 & arXiv links coming soon!)\
Overview blog post: [Medium](https://medium.com/superb-ai/training-non-english-nlp-models-with-english-training-data-664bbd260681)

## ISR Encoder Training

Script for training the ISR Encoder. Requires monolingual corpora cache files for training.

Prerequisites:

The following cache files saved in the 'data_dir' directory:
- Monolingual corpora sentences cache files, as mc_##.npy (e.g. mc_en.npy) where ## corresponds to [ISO 639-1 Code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) of each 'train_languages'; refer to Parsing and Caching Scripts section below.
- (If do_mid_train_eval) XNLI dev examples cache file, as [DEV.npy](___); refer to Parsing and Caching Scripts section below.

(If do_mid_train_eval,) The following model files in the 'mid_train_eval_nli_model_path' directory (the trailing 'nli_solver' is model name and not part of the directory):
-  English NLI solver model files, as [nli_solver.meta](___), [nli_solver.index](___), and [nli_solver.data-00000-of-00001](___) (note that 'mid_train_eval_nli_target_language' should be fixed as English when using this NLI solver).
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
  --mid_train_eval_nli_model_path=nli_solver_path/nli_solver \
```

## Classifier Training

Code for training a classifier on top of fixed ISR Encoder. Requires NLI training examples (mostly available in high-resource language, i.e. English) for training.

Prerequisites:

The following cache files saved in the 'data_dir' directory:
- NLI training examples cache file(s), as bse_##.npy (e.g. bse_en.npy) where ## corresponds to [ISO 639-1 Code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) of each 'xnli_train_languages'; refer to Parsing and Caching Scripts section below. Theoretically, NLI training examples from multiple languages can be used jointly from training the classifier on top of ISR (while the underlying assumption is that only English training examples are widely available currently).
- (if do_mid_train_eval) XNLI dev examples cache file, as [DEV.npy](___).

The following files in the 'isr_encoder_dir' directory:
- The ISR Encoder model files outputted from the ISR Encoder Training section above, as isr_encoder.meta, isr_encoder.index, isr_encoder.data-00000-of-00001. Alternatively, the ISR Encoder trained during our study can downloaded here: [isr_encoder.meta](___), [isr_encoder.index](___), and [isr_encoder.data-00000-of-00001](___).
- The language reference file, as language_reference.json. The language reference file corresponding to our study's ISR Encoder can be downloaded here: [language_refernce.json](___)

```
python train_classifier.py \
  --data_dir=data \
  --isr_encoder_dir=isr_encoder_dir \
  --isr_encoder_name=isr_encoder \
  --output_dir=outputs/model_isrFixed_trISRfromXNLIEn42 \
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

## Parsing and Caching Scripts

### Producing a monolingual corpora cache file from a Wikipedia dump

1. Download the [Wikipedia dump](https://dumps.wikimedia.org/) of the language of interest (.XML file).

2. Use [WikiExtractor](https://github.com/attardi/wikiextractor) to extract and clean text from the XML file, outputting a file (e.g. wiki_00) in the "AA" folder within the 'output' directory. The "100G" 'bytes' parameter in our sample usage is to ensure that only 1 file is outputted (rather than broken up into multiple):

Prerequisites:
- The downloaded dump file (e.g. en_dump.xml) in the current directory.
```
python WikiExtractor.py \
 --output=en_extracted \
 --bytes=100G \
en_dump.xml
```

3. Run mc_custom_extraction.py on once-extracted file to perform custom extraction and cleanup to output a .txt file.

Prerequisites:
- The once-extracted dump file (e.g. wiki_00) in the 'source_file_path' directory (the trailing source file name is not part of the directory).

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
[mc_en.txt](___)\
[mc_es.txt](___)\
[mc_de.txt](___)\
[mc_zh.txt](___)\
[mc_ar.txt](___)

3. Run FILENAME.py (caching script) to produce cache files.

The monolingual corpora cache files used in our study can be downloaded here:\
[mc_en.npy](___)\
[mc_es.npy](___)\
[mc_de.npy](___)\
[mc_zh.npy](___)\
[mc_ar.npy](___)

### Producing a NLI examples cache file

1. Download the XNLI 



The NLI examples cache files used in our study can be downloaded here:\
[bse_en.npy](___)\
[DEV.npy](___)\
[TEST.npy](___)


## Paper Abstract

As numerous modern NLP models demonstrate high-performance in various tasks when trained with resource-rich language data sets such as those of English, there has been a shift in attention to the idea of applying such learning to low-resource languages via zero-shot or few-shot cross-lingual transfer. While the most prominent efforts made previously on achieving this feat entails the use of parallel corpora for sentence alignment training, we seek to generalize further by assuming plausible scenarios in which such parallel data sets are unavailable. In this work, we present a novel architecture for training interlingual semantic representations on top of sentence embeddings in a completely unsupervised manner, and demonstrate its effectiveness in zero-shot cross-lingual transfer in natural language inference task. Furthermore, we showcase a method of leveraging this framework in a few-shot scenario, and finally analyze the distributional and permutational alignment across languages of these interlingual semantic representations.

## ISR Encoder Training Framework

<img src="https://github.com/ChannyHong/ISREncoder/blob/master/imgs/training_flow.png" width="900px"/>

The GAN-based framework consists of a single Generator — composed of an encoder (to be our ISR Encoder) and a decoder — that generates sentences in target domains given real sentences in their original domains. Within the same training scheme, we jointly train a Discriminator that performs the following two tasks: distinguishing whether a given sentence is real or generated & classifying the domain of a given sentence (both real and generated).

## Main Results

Model Type | en | es | de | zh | ar
---------- | :------: | :------: | :------: | :------: | :------:
BSE (Baseline) | 63.8 | 57.1 | 51.9 | 53.4 | 50.2
ISR (λisr = 0) | 65.2 | 57.9 | 55.0 | 55.8 | 50.4
ISR (λcls = 0) | 60.1 | 56.1 | 52.6 | 51.2 | 50.0
ISR (λrec = 0) | 37.6 | 36.3 | 36.0 | 38.0 | 37.4
**ISR** | **65.4** | **60.4** | **58.8** | **58.4** | **55.4**
   

