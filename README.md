# NTUADL2020

Applied Deep Learning (2020 Spring) @ NTU

[Course website link](https://www.csie.ntu.edu.tw/~miulab/s108-adl/syllabus?fbclid=IwAR1n5ldKrapBjMc6JV0uUkzU52SSzquBjk1cwh6kPHkS4v0d7TdPiyJm_f4)

# Assignment

  ##### 0. NLP FROM SCRATCH: CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN
  
  ##### 1. [Summarization website](https://www.csie.ntu.edu.tw/~miulab/s108-adl/A1)
  	
   - architecture:
    		
         use BiLSTM  -> summary extraction
         use seq2seq -> summary abstraction (encoder, decoder implement)
         use seq2seq + attention -> summary abstraction -> (encoder, decoder, attention)
     
   - data: please download [here](https://drive.google.com/drive/folders/1L_ayPqKlm6KmimjTHvheLQgm2EZfajh4)
      	
   - Data Processing
     
     	 1. load pretrained embedding from Glove
         2. Tokenization with keras tokenizer 
            - Fit on text
            - text to sequence
            - pad sequence
          
   - Training
   
     * BiLSTM for Summary Extraction

     * seq2seq for summary abstraction

     * seq2seq + attention for summary abstraction
      		

           
     
           
         
         
  
     
   
  
