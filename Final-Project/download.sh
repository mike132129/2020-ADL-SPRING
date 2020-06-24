#!/usr/bin/evn bash
# download model to predict
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nQyU5fX4WpcwFzHTfN8zpMvjkEWUKkr_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nQyU5fX4WpcwFzHTfN8zpMvjkEWUKkr_" -O ./model/bert_final.pth && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZM-zqLDNbMwHjZ3-MBhXJiCWG8Osk-ax' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZM-zqLDNbMwHjZ3-MBhXJiCWG8Osk-ax" -O ./model/bert-bilstm-crf.pth && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zvq1QLh5c-eC1tN-cd4phJgNCfIPjCse' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zvq1QLh5c-eC1tN-cd4phJgNCfIPjCse" -O ./embedding/model/bert0621-epoch-3/pytorch_model.bin && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18FEjqflQu9dPWWVJytR5YL0s0U7qLCo1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18FEjqflQu9dPWWVJytR5YL0s0U7qLCo1" -O ./embedding/adl-pretrained-model/bert-embedding-epoch-9/pytorch_model.bin && rm -rf /tmp/cookies.txt


