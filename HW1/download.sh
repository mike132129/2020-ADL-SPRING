#!/usr/bin/evn bash
abstractive_with_att=https://www.dropbox.com/s/rpt1ylq1o3y73r6/abstractive_with_att.hdf5?dl=1
abstractive_without_att=https://www.dropbox.com/s/08z50f8whzw3l8t/abstractive_without_att.hdf5?dl=1
extractive=https://www.dropbox.com/s/mykm3utckn9h0tz/extractive.hdf5?dl=1
text_embedding=https://www.dropbox.com/s/rz9ticx5fv5dm69/text_embedding.npy?dl=1
X_tokenizer=https://www.dropbox.com/s/txp030gcm17rpfc/X_tokenizer.pkl?dl=1
Y_tokenizer=https://www.dropbox.com/s/q3zws32vpijw363/Y_tokenizer.pkl?dl=1

# download pretrained embedding matrix
wget "${text_embedding}" -O ./data/text_embedding.npy

# download model
wget "${extractive}" -O ./model/extractive.hdf5
wget "${abstractive_without_att}" -O ./model/abstractive_without_att.hdf5
wget "${abstractive_with_att}" -O ./model/abstractive_with_att.hdf5

# Load tokenizer
wget "${X_tokenizer}" -O ./data/X_tokenizer.pkl
wget "${Y_tokenizer}" -O ./data/Y_tokenizer.pkl


