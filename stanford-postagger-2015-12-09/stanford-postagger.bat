:: usage: stanford-postagger model textFile
::  e.g., stanford-postagger models\english-left3words-distsim.tagger sample-input.txt

java -mx300m -classpath stanford-postagger.jar;lib\* edu.stanford.nlp.tagger.maxent.MaxentTagger -model models\english-left3words-distsim.tagger -textFile temp.txt > tagged_temp.txt
