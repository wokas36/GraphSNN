# A New Perspective on "How Graph Neural Networks Go Beyond Weisfeiler-Lehman?"

* This is a Python3 implementation of generalised message-passing framework (GMP) and GraphSNN neural model for node classification and graph classification.

### Prerequisites

* torch (>= 1.7.1)
* keras (>= 2.2.2)
* tensorflow (>= 1.9.0)
* torch_geometric (>= 2.0.1)
* ogb (>= 1.3.1)
* networkx (>= 2.2)
* numpy (>=1.19.0)
* scipy (>=1.2.1)
* cython (>=0.27.3)

### Evaluation and dataset references for node classification

	* Files description
		* graphsn_standard_splits_node_classification.ipynb - node classification with standard splits (ipython notebook version)
		* graphsn_random_splits_node_classification.ipynb - node classification with random splits (ipython notebook version)
		* utils.py - Data preprocessing and loading the data
		* models.py - n-layer GNN model with GraphSNN_M for node classification
		* layers.py - GraphSNN_M layer

	* For node classification tasks, we use four citation network datasets:
		[1] Cora, Citeseer and Pubmed for semi-supervised document classification
		[2] NELL for semi-supervised entity classification

	* The experimental results show that our method consistently outperforms all state-of-the-art methods on all benchmark datasets
	* We consider the four popular message-passing GNNs: Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), Graph Isomorphism Network (GIN), and GraphSAGE.
    
### Evaluation and dataset references for large graph classification (OGB graph dataset)

	* Files description
		* ogbg_mol.ipynb - GraphSNN evaluation on OGB graph dataset (ipython notebook version)
		* conv.py - GraphSNN convolution along the graph structure
		* gnn.py - GraphSNN pooling function to generate whole-graph embeddings
        
	* For large graph classification tasks, we use five large graph datasets from Open Graph Benchmark (OGB), including four molecular graph datasets (ogbg-molhiv, ogbg-moltox21, ogbg-moltoxcast and ogb-molpcba) and one protein-protein association network (ogbg-ppa). we also consider a variant, denoted as GraphSNN+VN, which performs the message passing over augmented graphs with virtual nodes in GraphSNN.

### Evaluation and dataset references for small graph classification
    
	* Files description
		* graphsn_graph_classification.ipynb - GraphSNN cross validation (ipython notebook version)
		* graph_data.py - Data preprocessing and loading the data
		* data_reader.py - Read the txt files containing all data of the dataset
		* models.py - GNN model with multiple GraphSNN layers for constructing the readout function
		* layers.py - GraphSNN layer
        
	* We evaluate our model on standard stratified splits and random splits. We use eight benchmark datasets grouped in two categories. You can find all datasets for graph classification in data folder [path: Graph_Classification/data/].
		(1) 6 from bioinformatics datasets - MUTAG, PTC-MR, COX2, BZR, PROTEINS, and D&D
		(2) 2 from social network datasets - IMDB-B and RDT-M5K

	* The experimental results show that our method consistently outperforms all state-of-the-art methods on all benchmark datasets
	* We provided subset of our datasets and we will release the full datasets and hyperparameter settings after acceptance.
	
## Citation

Please cite our paper if you use this code in your research work.

```
@inproceedings{wijesinghe2021iclr,
  title={A New Perspective on" How Graph Neural Networks Go Beyond Weisfeiler-Lehman?"},
  author={Wijesinghe, Asiri and Wang, Qing},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

## License

MIT License

## Contact for DFNets issues
Please contact me: asiri.wijesinghe@anu.edu.au if you have any questions / submit a Github issue if you find any bugs.