
# KR-BERT-KOSAC


A pretrained Korean-specific BERT model including sentiment features to perform better at sentiment-related tasks, developed by Computational Linguistics Lab at Seoul National University.

It is based on our character-level [KR-BERT](https://github.com/snunlp/KR-BERT) models which utilize WordPiece and BidirectionalWordPiece tokenizers.

<br>

## Sentiment Features

We use the predefined sentiment lexicon of the Korean Sentiment Analysis Corpus ([KOSAC](http://ling.snu.ac.kr/kosac/)) to construct sentiment features. The corpus contains 17,582 annotated sentiment expressions from 332 documents and 7,744 sentences from Sejong Corpus and news articles. The sentiment expressions include values of subjectivity, polarity, intensity, manner of expressions, etc.

The sentiment features included in KOSAC contain polarity and intensity values that we use in our models. There are five classes of polarity values: None (no polarity value), POS (positive), NEUT (neutral), NEG (negative) and COMP (complex).

The four classes of intensity values include: None (no intensity value), High, Medium and Low. These values show how strong the sentiment is in the token.


![efg](./img/fig2_new.png)


The polarity and intensity embeddings can be simply added to the token, position and segment embeddings of BERT and be trained just as BERT models.


<br>

## Masked LM Accuracy
| Model                                 | MLM acc   |
|-------------------------------------- |---------  |
| KoBERT                                | 0.750     |
| KR-BERT WordPiece                     | 0.779     |
| KR-BERT BidirectionalWordPiece        | 0.769     |
| KR-BERT-KOSAC WordPiece               | 0.851     |
| KR-BERT-KOSAC BidirectionalWordPiece  | **0.855**     |

<br>

## Models
### tensorflow

* A model using BERT (WordPiece) tokenizer ([download](https://drive.google.com/file/d/1IXAUIDHzK9LN09AoG2SKFkTm3TyH5lrg/view?usp=sharing))
* A model using BidirectionalWordPiece tokenizer ([download](https://drive.google.com/file/d/17asdtEFSVz2UuH7cCxUWb6jK7GwEwDO9/view?usp=sharing))

<br>

## Downstream tasks

### Naver Sentiment Movie Corpus (NSMC)

* You can use the original BERT WordPiece tokenizer by entering `bert` for the `tokenizer` argument, and if you use `ranked` you can use our BidirectionalWordPiece tokenizer.

* Download the checkpoint model and enter its path to `init_checkpoint`.

* Download the [NSMC data](https://github.com/snunlp/KR-BERT/tree/master/krbert_tensorflow/data/char) and enter its path to `data_dir`.


```sh
# tensorflow

python3 run_classifier_kosac.py \
  --task_name=NSMC \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir={data_dir} \
  --tokenizer={bert, ranked} \
  --vocab_file=vocab_char_16424.txt \
  --bert_config_file=bert_config_char16424.json \
  --init_checkpoint={model_dir} \
  --do_lower_case=False\
  --max_seq_length=128 \
  --train_batch_size=128 \
  --learning_rate=5e-05 \
  --num_train_epochs=5.0 \
  --output_dir={output_dir}
 
```

<br>

#### NSMC Acc.


| Model                                 | eval acc  | test acc  |
|-------------------------------------- |---------- |---------- |
| multilingual BERT                     | 0.8708    | 0.8682    |
| KorBERT                               | 0.8556    | 0.8555    |
| KR-BERT WordPiece                     | 0.8986    | 0.8974    |
| KR-BERT BidirectionalWordPiece        | 0.9010    | 0.8954    |
| KR-BERT-KOSAC WordPiece               | **0.9030**    | **0.8982**    |
| KR-BERT-KOSAC BidirectionalWordPiece  | 0.902     | 0.896     |

<br>

## Contacts

nlp.snu@gmail.com

