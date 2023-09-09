Automated Learning of Event Schema from Text

# Requirements

Python 
Python packages
- PyTorch
- transformers
- tqdm
- lxml
- nltk


# Stepwise Process

## XML Parsing
The `XMLParsing.py` script extracts the text tags from the hievents dataset saved in the `hievents` directory and converts it to an csv file which has the article name and the text of each article. For the csv file, refer `hievents.csv`.

```
python XMLParsing.py
```

## XML to txt files
The `convertingToTxt.py` script extracts the text tags from the hievents dataset saved in the `hievents` directory and converts it to a text file with the article name as the file name and the text as its contents. For the text files, refer `hievents_txt` directory.

```
python convertingToTxt.py
```

Text Example:
```
A bus carrying Turkish pilgrims has come under attack in northern Syria, Turkish media and officials say.
```

## Entity Recognition, Relation Extraction and Event Extraction
For event recognition, relation extraction and event extraction, OneIE software was used. The software can be seen in the directory `oneie_v0.4.8`. The model for the same is `english.role.v0.3`. So, using the OneIE software, the entities, relations and events were predicted. The prediction was done with the help of `predict.py` script. The input for the prediction is `hievents_txt` directory and the outputs is saved in `output` directory.

```
python oneie_v0.4.8/oneie_v0.4.8/predict.py  -m "english.role.v0.3.mdl" -i "hievents_txt" -o "output" --gpu --format txt
```

Arguments:
- -i, --input: Path to the input directory 
- -o, --output: Path to the output directory.
- -gpu, --gpu: To use gpu
- -format, --format: To specify that the input format will be text files in this case

The code for the same was ran in `OneIEprediction.ipynb`.

### Output Format

OneIE save results in JSON format. Each line is a JSON object for a sentence 
containing the following fields:
+ doc_id (string): Document ID
+ sent_id (string): Sentence ID
+ tokens (list): A list of tokens
+ token_ids (list): A list of token IDs (doc_id:start_offset-end_offset)
+ graph (object): Information graph predicted by the model
  - entities (list): A list of predicted entities. Each item in the list has exactly
  four values: start_token_index, end_token_index, entity_type, mention_type, score.
  For example, "[3, 5, "GPE", "NAM", 1.0]" means the index of the start token is 3, 
  index of the end token is 4 (5 - 1), entity type is GPE, mention type is NAM,
  and local score is 1.0.
  - triggers (list): A list of predicted triggers. It is similar to `entities`, while
  each item has three values: start_token_index, end_token_index, event_type, score.
  - relations (list): A list of predicted relations. Each item in the list has
  three values: arg1_entity_index, arg2_entity_index, relation_type, score.
  In the following example, `[1, 0, "ORG-AFF", 0.52]` means there is a ORG-AFF relation
  between entity 1 ("leader") and entity 0 ("North Korean") with a local
  score of 0.52.
  The order of arg1 and arg2 can be ignored for "SOC-PER" as this relation is 
  symmetric.
  - roles (list): A list of predicted argument roles. Each item has three values:
  trigger_index, entity_index, role, score.
  In the following example, `[0, 2, "Attacker", 0.8]` means entity 2 (Kim Jong Un) is
  the Attacker argument of event 0 ("detonate": Conflict:Attack), and the local
  score is 0.8.

### Output Example: 
```
{"doc_id": "article-1126.txt", "sent_id": "article-1126.txt-0", "token_ids": ["article-1126.txt:0-1", "article-1126.txt:1-2", "article-1126.txt:2-3", "article-1126.txt:3-4", "article-1126.txt:4-5", "article-1126.txt:5-6", "article-1126.txt:6-7", "article-1126.txt:7-8", "article-1126.txt:8-9", "article-1126.txt:9-10", "article-1126.txt:10-11", "article-1126.txt:11-12", "article-1126.txt:12-13", "article-1126.txt:13-14", "article-1126.txt:14-15", "article-1126.txt:15-16", "article-1126.txt:16-17", "article-1126.txt:17-18", "article-1126.txt:18-19"], "tokens": ["A", "bus", "carrying", "Turkish", "pilgrims", "has", "come", "under", "attack", "in", "northern", "Syria", ",", "Turkish", "media", "and", "officials", "say", "."], "graph": {"entities": [[1, 2, "VEH", "NOM", 1.0], [3, 4, "GPE", "NAM", 0.41643227472701866], [4, 5, "PER", "NOM", 1.0], [11, 12, "LOC", "NAM", 0.5626992894494245], [13, 14, "GPE", "NAM", 1.0], [14, 15, "ORG", "NOM", 0.5626992894494245], [16, 17, "PER", "NOM", 1.0]], "triggers": [[8, 9, "Conflict:Attack", 1.0]], "relations": [[2, 1, "GEN-AFF", 0.6285213093501514], [5, 4, "GEN-AFF", 0.6285213093501514], [6, 4, "ORG-AFF", 1.0]], "roles": [[0, 0, "Target", 1.0], [0, 3, "Place", 1.0]]}}
```

## Converting OneIE to PathLM format
The output from the OneIE prediction that is in the `output` directory is converted to the PathLM format using the `ConvertingToPathLMJson.ipynb`. The converted format can be seen in the `pathlm_input` directory. 

### Output Format:
+ doc_id (string): Document ID
+ sent_id (string): Sentence ID
+ tokens (list): A list of tokens
+ pieces(list): A list of token pieces
+ token_lens (list): A list of number of pieces in a token
+ sentence(string): A string of the input sentence
+ entity_mentions(list): 
  - id (string) : Entity/Token ID
  - text(string) : Entity text
  - entity_type(string) : Entity Type
  - mention_type(string) : Entity Mention Type
  - start(int) : Starting index of entity
  - end(int) : Ending index of Entity

+ relation_mentions(list): 
  - id (string) : Relation/Token ID
  - relation_type(string) :Relation Type
  - arguments(list):
    Item-1
    - entity_id (string) : Entity ID of Argument-1
    - text(string) : Entity text of Argument-1
    - role(string) : Argument number
    Item-2
    - entity_id (string) : Entity ID of Argument-2
    - text(string) : Entity text of Argument-2
    - role(string) : Argument number

+ event_mentions(list):
  - id (string) : Event/Token ID
  - relation_type(string) :Event Type
  - trigger(dict) : 
    - text(string) : Event Name
    - start(int) : Event Start Index
    - end(int) : Event End Index
  - arguments(list):
    Item-1
    - entity_id (string) : Entity ID of Argument-1
    - text(string) : Entity text of Argument-1
    - role(string) : Entity Role
    Item-2
    - entity_id (string) : Entity ID of Argument-2
    - text(string) : Entity text of Argument-2
    - role(string) : Entity Role


### Output Example:
```
{"doc_id": "article-1126.txt", "sent_id": "article-1126.txt-0", "tokens": ["A", "bus", "carrying", "Turkish", "pilgrims", "has", "come", "under", "attack", "in", "northern", "Syria", ",", "Turkish", "media", "and", "officials", "say", "."], "pieces": ["a", "bus", "carrying", "turkish", "pilgrims", "has", "come", "under", "attack", "in", "northern", "syria", ",", "turkish", "media", "and", "officials", "say", "."], "token_lens": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "sentence": "A bus carrying Turkish pilgrims has come under attack in northern Syria, Turkish media and officials say.", "entity_mentions": [{"id": "article-1126.txt-E1-1", "text": "bus", "entity_type": "VEH", "mention_type": "NOM", "start": 1, "end": 2}, {"id": "article-1126.txt-E1-2", "text": "Turkish", "entity_type": "GPE", "mention_type": "NAM", "start": 3, "end": 4}, {"id": "article-1126.txt-E1-3", "text": "pilgrims", "entity_type": "PER", "mention_type": "NOM", "start": 4, "end": 5}, {"id": "article-1126.txt-E1-4", "text": "Syria", "entity_type": "LOC", "mention_type": "NAM", "start": 11, "end": 12}, {"id": "article-1126.txt-E1-5", "text": "Turkish", "entity_type": "GPE", "mention_type": "NAM", "start": 13, "end": 14}, {"id": "article-1126.txt-E1-6", "text": "media", "entity_type": "ORG", "mention_type": "NOM", "start": 14, "end": 15}, {"id": "article-1126.txt-E1-7", "text": "officials", "entity_type": "PER", "mention_type": "NOM", "start": 16, "end": 17}], "relation_mentions": [{"id": "article-1126.txt-R1-1", "relation_type": "GEN-AFF", "arguments": [{"entity_id": "article-1126.txt-E1-3", "text": "pilgrims", "role": "Arg-1"}, {"entity_id": "article-1126.txt-E1-2", "text": "Turkish", "role": "Arg-2"}]}, {"id": "article-1126.txt-R1-2", "relation_type": "GEN-AFF", "arguments": [{"entity_id": "article-1126.txt-E1-6", "text": "media", "role": "Arg-1"}, {"entity_id": "article-1126.txt-E1-5", "text": "Turkish", "role": "Arg-2"}]}, {"id": "article-1126.txt-R1-3", "relation_type": "ORG-AFF", "arguments": [{"entity_id": "article-1126.txt-E1-7", "text": "officials", "role": "Arg-1"}, {"entity_id": "article-1126.txt-E1-5", "text": "Turkish", "role": "Arg-2"}]}], "event_mentions": [{"id": "article-1126.txt-EV1-0", "event_type": "Conflict:Attack", "trigger": {"text": "attack", "start": 8, "end": 9}, "arguments": [{"entity_id": "article-1126.txt-E1-1", "text": "bus", "role": "Target"}, {"entity_id": "article-1126.txt-E1-4", "text": "Syria", "role": "Place"}]}]}
```

## Testing PathLM
The `pathlm_input` data is put into the a directory `pathlm_schema-master\pathlm_schema-master\data\ace ` and the the model is trained.

Step 1:
```
python pathlm_schema-master/pathlm_schema-master/data_utils/preprocessors/ace/path_discover.py
```

Step 2:
```
python pathlm_schema-master/pathlm_schema-master/data_utils/preprocessors/ace/path_tsv_vocab.py
```

Step 3:
```
bash pathlm_schema-master/pathlm_schema-master/path_xlnet_ft_clm.sh
```

Step 4:
```
python pathlm_schema-master/pathlm_schema-master/data_utils/preprocessors/ace/evaluate_path.py
```

The running of the same can be seen in the `TestingPathlm.ipynb`.

## Output 
The output files `pathlm_schema-master\pathlm_schema-master\data\ace\train.edges.direct.json` is a json file that consists of all the edges of the Event Schema Graph.

